#!/bin/bash -eu

SCRIPT_PATH="../utilities"

DATA=../data
RAW=${DATA}/raw
TOK=${DATA}/bpe/tok
mkdir -p ${TOK}

python ${SCRIPT_PATH}/text_processing.py -s train -j ${RAW}/vatex_training_v1.0.json -o ${TOK}/train -t tok
python ${SCRIPT_PATH}/text_processing.py -s valid -j ${RAW}/vatex_validation_v1.0.json -o ${TOK}/val -t tok
python ${SCRIPT_PATH}/text_processing.py -s test -j ${RAW}/vatex_public_test_english_v1.1.json -o ${TOK}/public_test -t tok -l en