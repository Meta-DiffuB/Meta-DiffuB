#!/bin/bash

DIFFUSION_STEPS=2000

NOISE_SCHEDULE="sqrt"
RESCALING_FACTOR=3.5

LR="5e-4"
MAX_TOKENS=12000 #16000
MAX_UPDATE=2000 #300000

DATASET="wmt14/wmt14.en-de.dist.bin"
# MODEL_DIR="models/${DATASET}"
MODEL_DIR="models/Meta_dinoiser_wmt14_en_de/"

mkdir -p $MODEL_DIR/tb

CUDA_VISIBLE_DEVICES=0 fairseq-train \
    ${DATASET} \
    --save-dir $MODEL_DIR \
    --ddp-backend no_c10d \
    --user-dir difformer \
    --task difformer \
    --criterion meta_diffuseq_loss \
    --arch difformer \
    --share-all-embeddings \
    --diffusion-steps $DIFFUSION_STEPS \
    --noise-schedule $NOISE_SCHEDULE --rescaling-factor $RESCALING_FACTOR \
    --embed-norm --embed-norm-affine \
    --self-cond \
    --rescale-timesteps \
    --optimizer adam --adam-betas '(0.9,0.98)' \
    --lr $LR --lr-scheduler inverse_sqrt \
    --min-lr '1e-09' --warmup-updates 10000 \
    --warmup-init-lr '1e-07' --label-smoothing 0.1 \
    --clip-norm 1 \
    --dropout 0.1 --weight-decay 0.01 \
    --decoder-learned-pos \
    --encoder-learned-pos \
    --pred-length-offset \
    --length-loss-factor 0.1 \
    --apply-bert-init \
    --fp16 \
    --log-format 'json' --log-interval 100 \
    --tensorboard-logdir $MODEL_DIR/tb \
    --fixed-validation-seed 7 \
    --decoding-steps 20 \
    --eval-bleu \
    --eval-tokenized-bleu \
    --eval-bleu-remove-bpe \
    --validate-interval 5 \
    --best-checkpoint-metric bleu \
    --maximize-best-checkpoint-metric \
    --max-tokens $MAX_TOKENS \
    --max-update $MAX_UPDATE \
    --num-workers 20 \
    --keep-last-epochs 10 \
    --save-interval-updates 100 \
    --keep-best-checkpoints 5 \
    --reset-optimizer \
    2>&1 | tee $MODEL_DIR/train.log
