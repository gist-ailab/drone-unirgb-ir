#!/bin/bash
set -e

ENV_NAME="unirgb-ir2"
source ~/anaconda3/etc/profile.d/conda.sh
conda activate $ENV_NAME

CUDA_VISIBLE_DEVICES=0 \
python tools/train.py \
configs/_vpt_cascade-rcnn/FLIR_RGBT_ViTDet/100ep/backbone_vitb_IN1k_mae_coco_224x224/1024_v15_rgb-ir-feat-fusion-spm_crossAttn-GRU_staged_8x1bs.py \
--work-dir work_dirs/1024_v15_rgb-ir-feat-fusion-spm_crossAttn-GRU_staged_8x1bs
