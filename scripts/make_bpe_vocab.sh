#!/bin/bash

DATA=../data/bpe/
TOK=${DATA}/tok_color_deprivation
mkdir ${TOK}
OUT=${TOK}/vocab
mkdir -p ${OUT}

nmtpy-build-vocab ${TOK}/en/8000/train.min-0.en  -o ${OUT}
nmtpy-build-vocab ${TOK}/train.zh  -o ${OUT}

DATA=../data/bpe/
TOK=${DATA}/tok_noun_masking
mkdir ${TOK}
OUT=${TOK}/vocab
mkdir -p ${OUT}

nmtpy-build-vocab ${TOK}/en/8000/train.min-0.en  -o ${OUT}
nmtpy-build-vocab ${TOK}/train.zh -o ${OUT}

DATA=../data/bpe/
TOK=${DATA}/tok_verb_masking
mkdir ${TOK}
OUT=${TOK}/vocab
mkdir -p ${OUT}

nmtpy-build-vocab ${TOK}/en/8000/train.min-0.en -o ${OUT}
nmtpy-build-vocab ${TOK}/train.zh -o ${OUT}

DATA=../data/bpe/
TOK=${DATA}/tok/
mkdir ${TOK}
OUT=${TOK}/vocab

mkdir -p ${OUT}

nmtpy-build-vocab ${TOK}/en/8000/train.min-0.en -o ${OUT}
nmtpy-build-vocab ${TOK}/train.zh -o ${OUT}
