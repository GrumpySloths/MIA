# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for DiffPure. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import argparse
import logging
import yaml
import os
import time

import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from stadv_eot.attacks import StAdvAttack

import utils
from utils import str2bool, get_accuracy, get_image_classifier, load_data
from runners.diffpure_sde import RevGuidedDiffusion
from guided_diffusion.image_datasets import load_data
import torchvision.utils as tvu

import PIL.Image
import torch
from PIL import Image



class SDE_Adv_Model(nn.Module):
    def __init__(self, args, config):
        super().__init__()
        self.args = args

        # image classifier
        self.classifier = get_image_classifier(args.classifier_name).to(config.device)

        # diffusion model
        print(f'diffusion_type: {args.diffusion_type}')
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

    def forward(self, x, img_idx, it):
        start_time = time.time()
        x_re = self.runner.image_editing_sample_o(x, img_idx=img_idx, it=it)
        return x_re
    
    def update(self, x_start, x_t):
        self.runner.update(x_start, x_t)


def eval_autoattack(args, config, model, x_start, adv_batch_size, log_dir, img_idx):
    ngpus = torch.cuda.device_count()
    '''
    origin_path = os.path.join(args.out_dir, str(img_idx) + '_origin.png')
    sample = ((x_start + 1) * 127.5).clamp(0, 255).to(torch.uint8)
    sample = sample.permute(0, 2, 3, 1)
    sample = sample[0].cpu().numpy()
    print(sample.shape)
    img = PIL.Image.fromarray(sample, origin_path)
    '''
    '''
    
    '''
    #x_t_origin = x_t.detach().cpu()
    x_start = x_start.to(config.device)
    #x_start = (x_start + 1) * 0.5
    x_t = (x_start + 1) * 0.5
    #x_t = x_start
    
    betas = model.runner.betas.detach().cpu()
    e = torch.randn_like(x_t)
    total_noise_levels = 250
    #print(f'total_noise_levels: {total_noise_levels}')
    a = (1 - betas).cumprod(dim=0)
    x_t = (x_t * a[total_noise_levels - 1].sqrt() + e * (1.0 - a[total_noise_levels - 1]).sqrt()).to(config.device)
    x_t_origin = x_t.detach().cpu()
    
    x_t.requires_grad_()
   
    eps = 0.1
    eps = torch.tensor(eps).to(config.device)    #学习率
    
    model.update(x_start, x_t)
        
    for it in range(0, 21):
        if it%5 == 0:
            eps /= 2
            
        with torch.enable_grad():
            x_0 = model(x_t, img_idx, it)  #获得逆扩散后的图片x_0
            loss_function = torch.nn.MSELoss(reduce=True, size_average=True)
            x0_loss = loss_function(x_0.float(), x_start.float())
            grad = torch.autograd.grad(x0_loss, [x_t])[0].detach()
            x_t = x_t - eps*grad.sign()
            x_t = torch.tensor(x_t).to(config.device).requires_grad_()
            
            xt_loss = loss_function(x_t.detach().cpu().float(), x_t_origin.float())
            
            print("Iter=" , it , ": ", "x0: ", x0_loss.item(), "xt: ",  xt_loss.item())
            
            

def robustness_eval(args, config):
    middle_name = '_'.join([args.diffusion_type, args.attack_version]) if args.attack_version in ['stadv', 'standard', 'rand'] \
        else '_'.join([args.diffusion_type, args.attack_version, args.attack_type])
    log_dir = os.path.join(args.image_folder, args.classifier_name, middle_name,
                           'seed' + str(args.seed), 'data' + str(args.data_seed))
    os.makedirs(log_dir, exist_ok=True)
    args.log_dir = log_dir
    logger = utils.Logger(file_name=f'{log_dir}/log.txt', file_mode="w+", should_flush=True)

    ngpus = torch.cuda.device_count()
    adv_batch_size = args.adv_batch_size * ngpus
    print(f'ngpus: {ngpus}, adv_batch_size: {adv_batch_size}')

    # load model
    print('starting the model and loader...')
    model = SDE_Adv_Model(args, config)
    if ngpus > 1:
        model = torch.nn.DataParallel(model)
    model = model.eval().to(config.device)

    # load data
    data = load_data(
        data_dir = args.data,
        batch_size = 1,
        image_size = 256,
        class_cond = False,
        #deterministic=True,
    )

    num = 200
    for img_idx in range(0, num):
        x_val, y_val = next(data)
        eval_autoattack(args, config, model, x_val, adv_batch_size, log_dir, img_idx)
    
    logger.close()


def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()['__doc__'])
    # diffusion models
    parser.add_argument('--config', type=str, required=True, help='Path to the config file')
    parser.add_argument('--data_seed', type=int, default=0, help='Random seed')
    parser.add_argument('--data', type=str, required=True, help='Path to data')
    parser.add_argument('--out_dir', type=str, required=True, help='Path to out_dir')
    parser.add_argument('--model', type=str, required=True, help='Path to model')
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

    args.image_folder = os.path.join(args.exp, args.image_folder)
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
    robustness_eval(args, config)

