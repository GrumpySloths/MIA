# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for DiffPure. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import os
import random
import numpy as np

import torch
import torchvision.utils as tvu
import torchsde

from guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults
from score_sde.losses import get_optimizer
from score_sde.models import utils as mutils
from score_sde.models.ema import ExponentialMovingAverage
from score_sde import sde_lib
from guided_diffusion.respace import space_timesteps
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
)

import PIL.Image
import torch
from PIL import Image


def _extract_into_tensor(arr_or_func, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array or a func.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    if callable(arr_or_func):
        res = arr_or_func(timesteps).float()
    else:
        res = arr_or_func.to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)


def restore_checkpoint(ckpt_dir, state, device):
    loaded_state = torch.load(ckpt_dir, map_location=device)
    state['optimizer'].load_state_dict(loaded_state['optimizer'])
    state['model'].load_state_dict(loaded_state['model'], strict=False)
    state['ema'].load_state_dict(loaded_state['ema'])
    state['step'] = loaded_state['step']

    
    
class RevVPSDE(torch.nn.Module):
    def __init__(self, model, diffusion, score_type='guided_diffusion', beta_min=0.1, beta_max=20, N=1000,
                 img_shape=(3, 256, 256), model_kwargs=None):
        """Construct a Variance Preserving SDE.
        Args:
          model: diffusion model
          score_type: [guided_diffusion, score_sde, ddpm]
          beta_min: value of beta(0)
          beta_max: value of beta(1)
        """
        super().__init__()
        #self.r
        self.model = model
        self.score_type = score_type
        self.model_kwargs = model_kwargs
        self.img_shape = img_shape

        self.beta_0 = beta_min
        self.beta_1 = beta_max
        self.N = N
        self.discrete_betas = torch.linspace(beta_min / N, beta_max / N, N)
        self.alphas = 1. - self.discrete_betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self._respace()
        
        self.noise_type = "diagonal"
        self.sde_type = "ito"
        
        
        self.diffusion = diffusion
        

    def _respace(self):
        self.use_timesteps=space_timesteps(self.N, "250")
        self.timestep_map = []
        self.original_num_steps = self.N
        last_alpha_cumprod = 1.0
        new_betas = []
        for i, alpha_cumprod in enumerate(self.alphas_cumprod):
            if i in self.use_timesteps:
                new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                last_alpha_cumprod = alpha_cumprod
                self.timestep_map.append(i)
        self.discrete_betas = torch.tensor(np.array(new_betas))
        self.alphas = 1. - self.discrete_betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_1m_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

        self.alphas_cumprod_cont = lambda t: torch.exp(-0.5 * (self.beta_1 - self.beta_0) * t**2 - self.beta_0 * t)
        self.sqrt_1m_alphas_cumprod_neg_recip_cont = lambda t: -1. / torch.sqrt(1. - self.alphas_cumprod_cont(t))

        
    def update(self, x_start, x_t):
        self.x_start = x_start
        self.x_t = x_t
        
        
    def _scale_timesteps(self, t):
        assert torch.all(t <= 1) and torch.all(t >= 0), f't has to be in [0, 1], but get {t} with shape {t.shape}'
        return (t.float() * self.N).long()

    def vpsde_fn(self, t, x):
        beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
        drift = -0.5 * beta_t[:, None] * x
        diffusion = torch.sqrt(beta_t)
        return drift, diffusion

    def rvpsde_fn(self, t, x, return_type='drift'):
        """Create the drift and diffusion functions for the reverse SDE"""
        drift, diffusion = self.vpsde_fn(t, x)

        if return_type == 'drift':

            assert x.ndim == 2 and np.prod(self.img_shape) == x.shape[1], x.shape
            x_img = x.view(-1, *self.img_shape)

            if self.score_type == 'guided_diffusion':
                # model output is epsilon
                if self.model_kwargs is None:
                    self.model_kwargs = {}

                disc_steps = self._scale_timesteps(t)  # (batch_size, ), from float in [0,1] to int in [0, 1000]
                print(disc_steps)
                model_output = self.model(x_img, disc_steps, **self.model_kwargs)
                
                terms = self.diffusion.attack_losses(model_output.detach().cpu(), self.x_start.detach().cpu(), self.x_t.detach().cpu(), disc_steps.detach().cpu())
                #print(terms)
                
                # with learned sigma, so model_output contains (mean, val)
                model_output, _ = torch.split(model_output, self.img_shape[0], dim=1)
                assert x_img.shape == model_output.shape, f'{x_img.shape}, {model_output.shape}'
                model_output = model_output.view(x.shape[0], -1)
                score = _extract_into_tensor(self.sqrt_1m_alphas_cumprod_neg_recip_cont, t, x.shape) * model_output

            elif self.score_type == 'score_sde':
                # model output is epsilon
                sde = sde_lib.VPSDE(beta_min=self.beta_0, beta_max=self.beta_1, N=self.N)
                score_fn = mutils.get_score_fn(sde, self.model, train=False, continuous=True)
                score = score_fn(x_img, t)
                assert x_img.shape == score.shape, f'{x_img.shape}, {score.shape}'
                score = score.view(x.shape[0], -1)

            else:
                raise NotImplementedError(f'Unknown score type in RevVPSDE: {self.score_type}!')

            drift = drift - diffusion[:, None] ** 2 * score
            return drift

        else:
            return diffusion

    def f(self, t, x):
        """Create the drift function -f(x, 1-t) (by t' = 1 - t)
            sdeint only support a 2D tensor (batch_size, c*h*w)
        """
        t = t.expand(x.shape[0])  # (batch_size, )
        drift = self.rvpsde_fn(1 - t, x, return_type='drift')
        assert drift.shape == x.shape
        return -drift

    def g(self, t, x):
        """Create the diffusion function g(1-t) (by t' = 1 - t)
            sdeint only support a 2D tensor (batch_size, c*h*w)
        """
        t = t.expand(x.shape[0])  # (batch_size, )
        diffusion = self.rvpsde_fn(1 - t, x, return_type='diffusion')
        assert diffusion.shape == (x.shape[0], )
        return diffusion[:, None].expand(x.shape)

    
    

class RevGuidedDiffusion(torch.nn.Module):
    def __init__(self, args, config, device=None):
        super().__init__()
        self.args = args
        self.config = config
        if device is None:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.device = device

        # load model
        if config.data.dataset == 'ImageNet':
            img_shape = (3, 256, 256)
            model_dir = 'pretrained/guided_diffusion'
            model_config = model_and_diffusion_defaults()
            model_config.update(vars(self.config.model))
            model, diffusion = create_model_and_diffusion(**model_config)
            model.load_state_dict(torch.load(args.model, map_location='cpu'))

            if model_config['use_fp16']:
                model.convert_to_fp16()

        elif config.data.dataset == 'CIFAR10':
            img_shape = (3, 32, 32)
            model_dir = 'pretrained/score_sde'
            print(f'model_config: {config}')
            model = mutils.create_model(config)

            optimizer = get_optimizer(config, model.parameters())
            ema = ExponentialMovingAverage(model.parameters(), decay=config.model.ema_rate)
            state = dict(step=0, optimizer=optimizer, model=model, ema=ema)
            restore_checkpoint(f'{model_dir}/checkpoint_8.pth', state, device)
            ema.copy_to(model.parameters())

        else:
            raise NotImplementedError(f'Unknown dataset {config.data.dataset}!')

        model.eval().to(self.device)

        self.model = model
        
        self.rev_vpsde = RevVPSDE(model=model, diffusion=diffusion, score_type=args.score_type, img_shape=img_shape,
                                  model_kwargs=None).to(self.device)
        self.betas = self.rev_vpsde.discrete_betas.float().to(self.device)

        #print(f't: {args.t}, rand_t: {args.rand_t}, t_delta: {args.t_delta}')
        #print(f'use_bm: {args.use_bm}')


    def get_model(self):
        return self.model
    
    
    def image_editing_sample(self, img, x_start, out_dir, e = None, bs_id=0, tag=None):
        assert isinstance(img, torch.Tensor)
        batch_size = img.shape[0]
        state_size = int(np.prod(img.shape[1:]))  # c*h*w
        out_dir = self.args.log_dir

        img = img.to(self.device)
        x0 = img
        
        out_dir = "/workspace/images/flower/test/"
        tvu.save_image((x0 + 1) * 0.5, os.path.join(out_dir, f'original_input.png'))
        '''
        if bs_id < 1:
            os.makedirs(out_dir, exist_ok=True)
            tvu.save_image((x0 + 1) * 0.5, os.path.join(out_dir, f'original_input.png'))
            
        '''

        xs = []
        for it in range(self.args.sample_step):
            total_noise_levels = self.args.t
            
            #x = x0 * a[total_noise_levels - 1].sqrt() + e * (1.0 - a[total_noise_levels - 1]).sqrt()
            #在参数中我们直接将扩散后的图片(img)输入，所以不需要在这里进行源代码的扩散步骤
            x = x0

            if bs_id < 1:
                tvu.save_image((x + 1) * 0.5, os.path.join(out_dir, f'init_{it}.png'))

            epsilon_dt0, epsilon_dt1 = 0, 1e-5
            t0, t1 = 1 - self.args.t * 1. / 1000 + epsilon_dt0, 1 - epsilon_dt1 
            t_size = 2
            _dt = 0.1
            ts = torch.linspace(t0, t1, t_size).to(self.device)
            
            x_ = x.view(batch_size, -1)  # (batch_size, state_size)
            
            xs_ = torchsde.sdeint_adjoint(self.rev_vpsde, x_, ts, method='euler', dt = _dt)
            
            x0 = xs_[-1].view(x.shape)  # (batch_size, c, h, w)

            xs.append(x0)
            
            tvu.save_image((x0 + 1) * 0.5, os.path.join(out_dir, str(4) +'_' + str(1) + '.png'))
        

        return torch.cat(xs, dim=0)
    
    
    def update(self, x_start, x_t):
        self.rev_vpsde.update(x_start, x_t)
        
    def image_editing_sample_o(self, img, bs_id=0, tag=None, img_idx=0, it=0):
        assert isinstance(img, torch.Tensor)
        batch_size = img.shape[0]
        state_size = int(np.prod(img.shape[1:]))  # c*h*w

        assert img.ndim == 4, img.ndim
        img = img.to(self.device)
        x = img

        out_dir = self.args.out_dir
        x0_path = os.path.join(out_dir, str(img_idx) + '_x0_' + str(it) + '.png')
        xt_path = os.path.join(out_dir, str(img_idx) + '_xt_' + str(it) + '.png')
        
        
        
        xs = []
        for it in range(self.args.sample_step):

            '''
            e = torch.randn_like(x).to(self.device)
            total_noise_levels = self.args.t + np.random.randint(-self.args.t_delta, self.args.t_delta)
            print(f'total_noise_levels: {total_noise_levels}')
            a = (1 - self.betas).cumprod(dim=0).to(self.device)
            x = x * a[total_noise_levels - 1].sqrt() + e * (1.0 - a[total_noise_levels - 1]).sqrt()
            '''
            #tvu.save_image((x + 1) * 0.5, xt_path)
            sample = ((x + 1) * 127.5).clamp(0, 255).to(torch.uint8)
            sample = sample.permute(0, 2, 3, 1)
            sample = sample[0].cpu().numpy()
            #print(sample.shape)
            img = PIL.Image.fromarray(sample, 'RGB').save(xt_path)
    
            epsilon_dt0, epsilon_dt1 = 0, 1e-5
            t0, t1 = 1 - self.args.t * 1. / 1000 + epsilon_dt0, 1 - epsilon_dt1
            t_size = 2
            ts = torch.linspace(t0, t1, t_size).to(self.device)

            x_ = x.view(batch_size, -1)  # (batch_size, state_size)
            if self.args.use_bm:
                bm = torchsde.BrownianInterval(t0=t0, t1=t1, size=(batch_size, state_size), device=self.device)
                xs_ = torchsde.sdeint_adjoint(self.rev_vpsde, x_, ts, method='euler', bm=bm)
            else:
                xs_ = torchsde.sdeint_adjoint(self.rev_vpsde, x_, ts, method='euler')
            x0 = xs_[-1].view(x.shape)  # (batch_size, c, h, w)

            sample = ((x0 + 1) * 127.5).clamp(0, 255).to(torch.uint8)
            sample = sample.permute(0, 2, 3, 1)
            sample = sample[0].cpu().numpy()
            #print(sample.shape)
            img = PIL.Image.fromarray(sample, 'RGB').save(x0_path)

            xs.append(x0)

        return torch.cat(xs, dim=0)
    
    
