#!/bin/bash

STEPS=10

LENGTH_BEAM=5
NOISE_BEAM=4
EPOCH="378"
UPDATE="2000"

DATASET="qt/data-bin"
# MODEL_DIR="models/${DATASET}_1e-4"
MODEL_DIR="models/Meta_qt"

OUTPUT_NAME="evaluate_step${STEPS}_beam${LENGTH_BEAM}x${NOISE_BEAM}_${UPDATE}"
OUTPUT_DIR=$MODEL_DIR/$OUTPUT_NAME

DEVICE=0

ROUGE_INSTALLED=$(pip list | grep files2rouge)

mkdir -p $OUTPUT_DIR/tmp

CUDA_VISIBLE_DEVICES=$DEVICE fairseq-generate \
    $DATASET \
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
    --batch-size 8 \
    > $OUTPUT_DIR/output.txt

# aggregate scores
# bleu
tail -n 1 $OUTPUT_DIR/output.txt >> $OUTPUT_DIR/scores.txt
echo >> $OUTPUT_DIR/scores.txt

# rouge
grep ^T $OUTPUT_DIR/output.txt \
    | cut -f2- \
    > $OUTPUT_DIR/tmp/output.tok.ref

grep ^H $OUTPUT_DIR/output.txt \
    | cut -f3- \
    > $OUTPUT_DIR/tmp/output.tok.sys

# if [ -n "$ROUGE_INSTALLED" ]; then
#     files2rouge $OUTPUT_DIR/tmp/output.tok.ref $OUTPUT_DIR/tmp/output.tok.sys \
#         | sed -n 14p \
#         >> $OUTPUT_DIR/scores.txt
#     echo >> $OUTPUT_DIR/scores.txt
# fi

# bert score
cat $OUTPUT_DIR/tmp/output.tok.ref \
    | sacremoses -l en -j 4 detokenize \
    > $OUTPUT_DIR/tmp/output.ref

cat $OUTPUT_DIR/tmp/output.tok.sys \
    | sacremoses -l en -j 4 detokenize \
    > $OUTPUT_DIR/tmp/output.sys

CUDA_VISIBLE_DEVICES=$DEVICE bert-score \
    -r $OUTPUT_DIR/tmp/output.ref \
    -c $OUTPUT_DIR/tmp/output.sys \
    --lang en --m "microsoft/deberta-xlarge-mnli" \
    >> $OUTPUT_DIR/scores.txt

# echo "Finished $OUTPUT_NAME. Scores:"
# cat $OUTPUT_DIR/scores.txt
