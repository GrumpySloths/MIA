# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for DiffPure. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import os

from runners.diffpure_sde import RevGuidedDiffusion
import pynvml
import argparse
import logging
import yaml
import time
import functools

import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from stadv_eot.attacks import StAdvAttack

import utils
from utils import str2bool, get_accuracy, load_data

from runners.diffpure_ddpm import Diffusion
from runners.diffpure_guided import GuidedDiffusion

from runners.diffpure_ode import OdeGuidedDiffusion
from runners.diffpure_ldsde import LDGuidedDiffusion

from PIL import Image
import random
import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms
from guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults

from guided_diffusion.image_datasets import load_data

from guided_diffusion.losses import discretized_gaussian_log_likelihood
from guided_diffusion.nn import mean_flat
import PIL.Image
import torch
from PIL import Image
import PIL


# diffusion model + classifier，要攻击的 adaptive model
class SDE_Adv_Model(nn.Module):
    def __init__(self, args, config):
        super().__init__()
        self.args = args

        # diffusion model
        #print(f'diffusion_type: {args.diffusion_type}')
        if args.diffusion_type == 'ddpm':
            self.runner = GuidedDiffusion(args, config, device=config.device)
        elif args.diffusion_type == 'sde':
            self.runner = RevGuidedDiffusion(args, config, device=config.device)
        elif args.diffusion_type == 'ode':
            self.runner = OdeGuidedDiffusion(args, config, device=config.device)
        elif args.diffusion_type == 'ldsde':
            self.runner = LDGuidedDiffusion(args, config, device=config.device)
        elif args.diffusion_type == 'celebahq-ddpm':
            self.runner = Diffusion(args, config, device=config.device)
        else:
            raise NotImplementedError('unknown diffusion type')

        self.register_buffer('counter', torch.zeros(1, device=config.device))
        self.tag = None
        
    def reset_counter(self):
        self.counter = torch.zeros(1, dtype=torch.int, device=config.device)

    def set_tag(self, tag=None):
        self.tag = tag

    def forward(self, x, x_start, noise):
        counter = self.counter.item()
        start_time = time.time()
        x_re, rets = self.runner.image_editing_sample((x - 0.5) * 2, x_start, e = noise, bs_id=counter, tag=self.tag)
        out = (x_re + 1) * 0.5

        return out, rets
    
    def get_model(self):
        return self.runner.get_model()


def eval_autoattack(args, config, model, x_val, y_val, adv_batch_size, log_dir, index, diffusion):
    ngpus = torch.cuda.device_count()

    # ---------------- apply the attack to sde_adv ----------------
    #print(f'apply the attack to sde_adv [{args.lp_norm}]...')
   
    x_start = x_val.to(config.device)
    

    noise = torch.randn_like(x_start).to(config.device)
    t = torch.tensor([200]).long().to(config.device)
    x_t = diffusion.q_sample(x_start, t, noise=noise)
    
    x_t.requires_grad_()
    
            # x[1] -> x[0] 的损失函数，计算现在求得的高斯分布落在x[0]的概率
    
    epochs = 10
    eps = 0.1
    eps = torch.tensor(eps).to(config.device)    #学习率
    
            
    root_log_dir = args.log_dir


    #diffusion = diffusion.to(config.device)
    
    for it in range(1, epochs + 1):
        log_dir = os.path.join(root_log_dir, "iter="+str(it))
        #os.makedirs(log_dir)
        #args.log_dir = log_dir
        
        if it%10 == 0:
            eps /= 2
            
        with torch.enable_grad():
            x_1, rets = model(x_t, x_start, None)  #获得逆扩散后的图片x_0
            
            #print(x0_loss)
            #print(torch.autograd.grad(x0_loss, [x_t]))
            #print(rets[-1].shape)

            #print(grad = torch.autograd.grad(rets[-1], [x_t]))
            #grad = torch.autograd.grad(rets[-1], [x_t]).detach()
            

            disc_steps = torch.tensor([0]).long().to(config.device)
            #disc_steps = self._scale_timesteps(0.1)  # (batch_size, ), from float in [0,1] to int in [0, 1000]
            model_kwargs = {}
            diffusion_model = model.get_model()
            model_output = diffusion_model(x_1, disc_steps, **model_kwargs)
            
            losses = diffusion.attack_losses(model_output, x_start, x_1, disc_steps)
            loss = losses['loss']
            vb = losses['vb']
            mse = losses["mse"]
            
            '''
            mean, var = torch.split(model_output, x_0.shape[1], dim=1)
            #print(model_output)
                
            #var = rets[-1].to(config.device)
            #var = var.unsqueeze(0)
 
            decoder_nll = -discretized_gaussian_log_likelihood(
                x_start, means=mean, log_scales=0.5 * var
            )
            decoder_nll = mean_flat(decoder_nll) / np.log(2.0)
        
            mse = mean_flat((mean - x_start) ** 2)
            loss = mse + decoder_nll
            '''
            
            '''
            loss_function = torch.nn.MSELoss(reduce=True, size_average=True)
            x0_loss = loss_function(model_output.float(), x_start.float())
            grad = torch.autograd.grad(x0_loss, [x_t])[0].detach()
            print(grad)
            '''
            '''---------------end-----------------'''
    
            grad = torch.autograd.grad(loss, [x_t])[0].detach()
            x_t = x_t - eps*grad.sign()
            x_t = torch.tensor(x_t).to(config.device).requires_grad_()
            
            print("Iter=" , it , ": ", loss.item(), vb.item(), mse.item())
            
        sample = x_t
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(torch.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample[0].detach().cpu().numpy()
        out_dir = "images/flower/test/"
        img = PIL.Image.fromarray(sample, 'RGB').save(out_dir + str(index) +'_' + str(it) + '.png')
            
            
                          

                
def eval_stadv(args, config, model, x_val, y_val, adv_batch_size, log_dir):
    return


def robustness_eval(args, config, img, t, i):

    args.t = t
   
    log_dir = os.path.join(args.image_folder, str(t))
    os.makedirs(log_dir, exist_ok=True)
    args.log_dir = log_dir
    logger = utils.Logger(file_name=f'{log_dir}/log.txt', file_mode="w+", should_flush=True)

    ngpus = torch.cuda.device_count()
    adv_batch_size = args.adv_batch_size
    #print(f'ngpus: {ngpus}, adv_batch_size: {adv_batch_size}')

    # load model
    #print('starting the model and loader...')
    model = SDE_Adv_Model(args, config)

    model = torch.nn.DataParallel(model)
    model = model.eval().to(config.device)
    model = model.module

    
    y_val = None
    # eval classifier and sde_adv against attacks
    if args.attack_version in ['standard', 'rand', 'custom']:
        eval_autoattack(args, config, model, img, y_val, adv_batch_size, log_dir, i)
    else:
        raise NotImplementedError(f'unknown attack_version: {args.attack_version}')

    logger.close()


def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()['__doc__'])
    # diffusion models
    parser.add_argument('--config', type=str, required=True, help='Path to the config file')
    parser.add_argument('--data', type=str, required=True, help='Path to data')
    parser.add_argument('--model', type=str, required=True, help='Path to model')
    parser.add_argument('--data_seed', type=int, default=0, help='Random seed')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed')
    parser.add_argument('--exp', type=str, default='exp', help='Path for saving running related data.')
    parser.add_argument('--verbose', type=str, default='info', help='Verbose level: info | debug | warning | critical')
    parser.add_argument('-i', '--image_folder', type=str, default='images', help="The folder name of samples")
    parser.add_argument('--ni', action='store_true', help="No interaction. Suitable for Slurm Job launcher")
    parser.add_argument('--sample_step', type=int, default=1, help='Total sampling steps')
    parser.add_argument('--t', type=int, default=400, help='Sampling noise scale')
    parser.add_argument('--t_delta', type=int, default=15, help='Perturbation range of sampling noise scale')
    parser.add_argument('--rand_t', type=str2bool, default=False, help='Decide if randomize sampling noise scale')
    parser.add_argument('--diffusion_type', type=str, default='ddpm', help='[ddpm, sde]')
    parser.add_argument('--score_type', type=str, default='guided_diffusion', help='[guided_diffusion, score_sde]')
    parser.add_argument('--eot_iter', type=int, default=20, help='only for rand version of autoattack')
    parser.add_argument('--use_bm', action='store_true', help='whether to use brownian motion')

    # LDSDE
    parser.add_argument('--sigma2', type=float, default=1e-3, help='LDSDE sigma2')
    parser.add_argument('--lambda_ld', type=float, default=1e-2, help='lambda_ld')
    parser.add_argument('--eta', type=float, default=5., help='LDSDE eta')
    parser.add_argument('--step_size', type=float, default=1e-2, help='step size for ODE Euler method')

    # adv
    parser.add_argument('--domain', type=str, default='celebahq', help='which domain: celebahq, cat, car, imagenet')
    parser.add_argument('--classifier_name', type=str, default='Eyeglasses', help='which classifier to use')
    parser.add_argument('--partition', type=str, default='val')
    parser.add_argument('--adv_batch_size', type=int, default=64)
    parser.add_argument('--attack_type', type=str, default='square')
    parser.add_argument('--lp_norm', type=str, default='Linf', choices=['Linf', 'L2'])
    parser.add_argument('--attack_version', type=str, default='custom')

    parser.add_argument('--num_sub', type=int, default=1000, help='imagenet subset')
    parser.add_argument('--adv_eps', type=float, default=0.07)
    # parser.add_argument('--gpu_ids', type=str, default='0')

    args = parser.parse_args()
    

    # parse config file
    with open(os.path.join('configs', args.config), 'r') as f:
        config = yaml.safe_load(f)
    new_config = utils.dict2namespace(config)

    level = getattr(logging, args.verbose.upper(), None)
    if not isinstance(level, int):
        raise ValueError('level {} not supported'.format(args.verbose))

    handler1 = logging.StreamHandler()
    formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
    handler1.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler1)
    logger.setLevel(level)

    args.image_folder = args.exp
    os.makedirs(args.image_folder, exist_ok=True)

    # add device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    logging.info("Using device: {}".format(device))
    new_config.device = device

    # set random seed
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.benchmark = True

    return args, new_config


if __name__ == '__main__':
    args, config = parse_args_and_config()
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    
    data = load_data(
        data_dir = args.data,
        batch_size = 1,
        image_size = 256,
        class_cond = False,
        #deterministic=True,
    )
    
    model_config = model_and_diffusion_defaults()
    _, diffusion = create_model_and_diffusion(**model_config)
    
    t = 200
    num = 100
    for i in range(4, num):
        print("-----------------------------------------image: " + str(i) + "---------------------------------------------")
        batch, cond = next(data)
        robustness_eval(args, config, batch, t, i, diffusion)           
  
