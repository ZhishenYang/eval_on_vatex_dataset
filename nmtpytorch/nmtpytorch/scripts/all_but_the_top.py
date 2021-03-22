#!/usr/bin/env python
# original all-but-the-top code:
# https://gist.github.com/lgalke/febaaa1313d9c11f3bc8240defed8390

import sys, os
import logging
import argparse
logger = logging.getLogger(__name__)

import numpy as np
from sklearn.decomposition import PCA
from gensim.models import KeyedVectors

def all_but_the_top(v, D):
    """
    Arguments:
        :v: word vectors of shape (n_words, n_dimensions)
        :D: number of principal components to subtract
    """
    # 1. Subtract mean vector
    v_tilde = v - np.mean(v, axis=0)
    # 2. Compute the first `D` principal components
    #    on centered embedding vectors
    u = PCA(n_components=D).fit(v_tilde).components_  # [D, emb_size]
    # Subtract first `D` principal components
    # [vocab_size, emb_size] @ [emb_size, D] @ [D, emb_size] -> [vocab_size, emb_size]
    return v_tilde - (v @ u.T @ u)  

# main
def apply_all_but_the_top(input_file: str, output_file: str, n_comp: int):
    kv = KeyedVectors.load_word2vec_format(input_file, 
        binary=input_file.endswith('.bin'),
        unicode_errors='ignore')
    
    # reuse kv to save memory space
    kv.vectors = all_but_the_top(kv.vectors, n_comp)
    kv.save_word2vec_format(output_file, binary=output_file.endswith('.bin'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, help="Path to input file in word2vec format")
    parser.add_argument("-o", "--output", required=True, help="Path to output file")
    parser.add_argument("-d", "--n-components", required=False, type=int, default=3, help="Num of PCA components to substruct.")

    args = parser.parse_args()
    logger.info(args)

    apply_all_but_the_top(args.input, args.output, args.n_components)
