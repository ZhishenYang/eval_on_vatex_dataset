#!/bin/bash -eu

SCRIPT_PATH="../utilities"
DATA=../data/
RAW=${DATA}/raw
TOK=${DATA}/bpe/tok_progressive_masking

mkdir -p ${TOK}

for n_t in 2 4 6 10 20 30 ; do
  mkdir -p ${TOK}/${n_t}/
  python ${SCRIPT_PATH}/text_processing.py -s train -j ${RAW}/vatex_training_v1.0.json -o ${TOK}/${n_t}/train  -t progress_masking -n ${n_t}
  python ${SCRIPT_PATH}/text_processing.py -s valid -j ${RAW}/vatex_validation_v1.0.json -o ${TOK}/${n_t}/val -t progress_masking -n ${n_t}
  python ${SCRIPT_PATH}/text_processing.py -s test -j ${RAW}/vatex_public_test_english_v1.1.json -o ${TOK}/${n_t}/public_test -t progress_masking -n ${n_t} -l en
done
