#!/bin/bash -eu

NUM_MERGE=
VOCAB_THRES=0
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    -s)
      NUM_MERGE="$2"; shift 2; ;;
    -m)
      VOCAB_THRES="$2"; shift 2; ;;
    *)
      echo "Error: Unknown option '$1'"; exit 1; ;;
  esac
done

[ -z $NUM_MERGE ] && echo "Usage: $0 -s num_merge [-m min_freq]" && exit 1

TOK_DIRs=("../data/bpe/tok/" "../data/bpe/tok_verb_masking/" "../data/bpe/tok_noun_masking/" "../data/bpe/tok_color_deprivation/")
BPE_DIRs=("../data/bpe/tok/en/${NUM_MERGE}" "../data/bpe/tok_verb_masking/en/${NUM_MERGE}" "../data/bpe/tok_noun_masking/en/${NUM_MERGE}" "../data/bpe/tok_color_deprivation/en/${NUM_MERGE}")

for i in "${!TOK_DIRs[@]}"; do
  TOK_DIR=${TOK_DIRs[i]}
  BPE_DIR=${BPE_DIRs[i]}

  [ ! -d ${BPE_DIR} ] && mkdir -p ${BPE_DIR}

  echo "Work on ${TOK_DIR}"
  echo "Save on ${BPE_DIR}"

  [ ! -f ${BPE_DIR}/vocab ] && \
    subword-nmt learn-joint-bpe-and-vocab \
      -s $NUM_MERGE \
      -o ${BPE_DIR}/codes \
      -i ${TOK_DIR}/train.en \
      --write-vocabulary ${BPE_DIR}/vocab

  for SPLIT in train val test; do
    IN_NAME="${TOK_DIR}/${SPLIT}.en"
    OUT_NAME="${BPE_DIR}/${SPLIT}.min-${VOCAB_THRES}.en"
    subword-nmt apply-bpe \
      -c ${BPE_DIR}/codes \
      --vocabulary-threshold ${VOCAB_THRES} \
      --vocabulary ${BPE_DIR}/vocab \
      <${IN_NAME} >${OUT_NAME} &
done
done
wait