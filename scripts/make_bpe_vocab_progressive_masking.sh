#!/bin/bash

for n_t in 2 4 6 10 20 30 ; do
  DATA=../data/bpe/
  TOK=${DATA}/tok_progressive_masking/${n_t}/
  OUT=${TOK}/vocab
  mkdir -p ${OUT}

  ../nmtpytorch/bin/nmtpy-build-vocab ${TOK}/en/8000/train.min-0.en  -o ${OUT}
  ../nmtpytorch/bin/nmtpy-build-vocab ${TOK}/train.${n_t}.zh  -o ${OUT}
done
