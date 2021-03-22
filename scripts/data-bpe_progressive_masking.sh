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


for n_t in 2 4 6 10 20 30; do
  TOK_DIR="../data/bpe/tok_progressive_masking/${n_t}/"
  BPE_DIR="../data/bpe/tok_progressive_masking/${n_t}/en/${NUM_MERGE}"

  echo "${TOK_DIR}"
  echo "${BPE_DIR}"

  [ ! -d ${BPE_DIR} ] && mkdir -p ${BPE_DIR}

  [ ! -f ${BPE_DIR}/vocab ] && \
  subword-nmt learn-joint-bpe-and-vocab \
  -s $NUM_MERGE \
  -o ${BPE_DIR}/codes \
  -i ${TOK_DIR}/train.${n_t}.en \
  --write-vocabulary ${BPE_DIR}/vocab

  for SPLIT in train val public_test; do
  IN_NAME="${TOK_DIR}/${SPLIT}.${n_t}.en"
  OUT_NAME="${BPE_DIR}/${SPLIT}.min-${VOCAB_THRES}.en"
  subword-nmt apply-bpe \
  -c ${BPE_DIR}/codes \
  --vocabulary-threshold ${VOCAB_THRES} \
  --vocabulary ${BPE_DIR}/vocab \
  <${IN_NAME} >${OUT_NAME} &
  done
done
wait
