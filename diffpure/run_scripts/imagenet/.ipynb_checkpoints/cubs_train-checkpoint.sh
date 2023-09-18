#!/usr/bin/env bash
cd ../..

SEED1=10
SEED2=10

for t in 500; do
  for adv_eps in 0.0157; do
    for seed in $SEED1; do
      for data_seed in $SEED2; do

        CUDA_VISIBLE_DEVICES=0 python eval_sde_adv.py --exp ./exp_results --config imagenet.yml \
          -i imagenet-robust_adv-$t-eps$adv_eps-4x4-bm0-t0-end1e-5-cont \
          --data "/data/CUB_200_2011/train_20" \
          --model "models/cubs/model006000.pt" \
          --out_dir "/workspace/images/cubs/train/" \
          --t $t \
          --adv_eps $adv_eps \
          --adv_batch_size 1 \
          --num_sub 1 \
          --domain imagenet \
          --classifier_name imagenet-resnet50 \
          --seed $seed \
          --data_seed $data_seed \
          --diffusion_type sde \
          --attack_version standard\
            
      done
    done
  done
done
