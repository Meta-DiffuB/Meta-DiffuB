#!/bin/bash

STEPS=10

LENGTH_BEAM=5
NOISE_BEAM=4
EPOCH="301"
UPDATE="100"

DATASET="wmt14/wmt14.en-de.dist.bin"
MODEL_DIR="models/MetaDiffuB_wmt14_en_de"

OUTPUT_NAME="evaluate_step${STEPS}_beam${LENGTH_BEAM}x${NOISE_BEAM}_${UPDATE}"
OUTPUT_DIR=$MODEL_DIR/$OUTPUT_NAME

mkdir -p $OUTPUT_DIR/tmp

CUDA_VISIBLE_DEVICES=0 fairseq-generate \
    ${DATASET} \
    --gen-subset test \
    --user-dir difformer \
    --task difformer \
    --path $MODEL_DIR/checkpoint_${EPOCH}_${UPDATE}.pt \
    --decoding-steps $STEPS \
    --decoding-early-stopping 5 \
    --length-beam-size $LENGTH_BEAM \
    --noise-beam-size $NOISE_BEAM \
    --ppl-mbr \
    --remove-bpe \
    --batch-size 32 \
    > $OUTPUT_DIR/output.txt

cat $OUTPUT_DIR/output.txt \
    | grep ^H \
    | cut -f3- \
    | perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' \
    > $OUTPUT_DIR/tmp/output.sys

cat $OUTPUT_DIR/output.txt \
    | grep ^T \
    | cut -f2- \
    | perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' \
    > $OUTPUT_DIR/tmp/output.ref

perl ~/mosesdecoder/scripts/generic/multi-bleu.perl $OUTPUT_DIR/tmp/output.ref \
    < $OUTPUT_DIR/tmp/output.sys \
    > $OUTPUT_DIR/score.txt
echo >> $OUTPUT_DIR/score.txt

# bleu
tail -n 1 $OUTPUT_DIR/output.txt >> $OUTPUT_DIR/score.txt
echo >> $OUTPUT_DIR/score.txt
    
# echo "Finished $OUTPUT_NAME. BLEU:"
# cat $OUTPUT_DIR/bleu.txt
