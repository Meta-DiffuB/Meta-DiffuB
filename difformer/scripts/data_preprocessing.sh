fairseq-preprocess \
    --source-lang en --target-lang de \
    --trainpref wmt14.en-de.dist.bin/train \
    --validpref wmt14.en-de.dist.bin/valid \
    --testpref wmt14.en-de.dist.bin/test \
    --destdir data-bin/wmt14_en_de \
    --joined-dictionary \
    --workers 20