#!/usr/bin/env python

import sys, os
import logging
import argparse
logger = logging.getLogger(__name__)

import numpy as np
import torch
from gensim.models import KeyedVectors
from tqdm import tqdm
import math

# def compute_centroid(key, values, k):
#     key = key.unsqueeze(0) if key.dim() == 1 else key
#     # 1. compute similarity matrix
#     sim_matrix = torch.functional.F.cosine_similarity(key, values)
#     # 2. select top-k NNs
#     top_k = sim_matrix.argsort(descending=True)[1:k+1]
#     # 3. compute the centroid
#     centroid = values[top_k].mean(dim=0)
#     return centroid

def localized_centering(v: np.ndarray, k: int, batch_size=1024):
    """
    Arguments:
        :v: word vectors of shape (n_words, n_dimensions)
        :k: number of nearest neighbors to subtract
    """
    # 0. use torch for large vocab
    v = torch.torch.from_numpy(v)
    
    # 1. compute similarity matrix and select top-k NNS by batch
    v_norm = (v / v.norm(dim=1)[:, None]).cuda()
    top_k = []
    for i in tqdm(range(math.ceil(v.shape[0] / batch_size))):
        a_v_norm = v_norm[i*batch_size:(i+1)*batch_size, :]
        a_sim_matrix = torch.mm(a_v_norm, v_norm.t())
        a_top_k = a_sim_matrix.argsort(descending=True)[:, 1:k+1]
        top_k.append(a_top_k.cpu())
        del a_v_norm
        del a_sim_matrix
        del a_top_k

    # 2. subtract centroid of top-k NNs
    centroids = v[torch.cat(top_k)].mean(dim=1)
    v = v - centroids

    return v.numpy()

# main
def apply_localized_centering(input_file: str, output_file: str, 
    k: int, batch_size: int):
    kv = KeyedVectors.load_word2vec_format(input_file, 
        binary=input_file.endswith('.bin'),
        unicode_errors='ignore')
    
    # reuse kv to save memory space
    kv.vectors = localized_centering(kv.vectors, k, batch_size=batch_size)
    kv.save_word2vec_format(output_file, binary=output_file.endswith('.bin'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, help="Path to input file in word2vec format")
    parser.add_argument("-o", "--output", required=True, help="Path to output file")
    parser.add_argument("-k", "--k-nearest-neighbors", required=False, 
        type=int, default=3, help="Num of nearest neighbors to substruct.")
    parser.add_argument("-b", "--batch-size", required=False, 
        type=int, default=256, help="Batch size to compute k-NN.")

    args = parser.parse_args()
    logger.info(args)

    apply_localized_centering(args.input, args.output, 
        args.k_nearest_neighbors, batch_size=args.batch_size)
