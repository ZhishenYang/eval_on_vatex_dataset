import logging

import torch
from torch import nn
import numpy as np
import math

from .transformer import Transformer
from ..layers import StackedMultimodalTransformerEncoder, TransformerDecoder

logger = logging.getLogger('nmtpytorch')

class VGTDeep(Transformer):
    def set_defaults(self):
        super().set_defaults()
        self.defaults.update({
            "feat_name": "feats",
            "feat_dim": "feat_dim",
        })

    def __init__(self, opts):
        super().__init__(opts)

        self.feat_name = self.opts.model['feat_name']

    def reset_parameters(self):
        self.enc.reset_parameters()
        self.dec.reset_parameters()

    def setup(self, is_train=True):
        self.enc = StackedMultimodalTransformerEncoder(
            n_vocab = self.n_src_vocab,
            model_dim = self.opts.model['model_dim'],
            feat_dim = self.opts.model['feat_dim'],
            n_heads = self.opts.model['n_heads'],
            key_size = self.opts.model['key_size'],
            inner_dim = self.opts.model['inner_dim'],
            n_layers = self.opts.model['n_layers'],
            max_len = self.opts.model['max_len'],
            dropout = self.opts.model['dropout'],
        )
        self.dec = TransformerDecoder(
            n_vocab = self.n_trg_vocab,
            model_dim = self.opts.model['model_dim'],
            n_heads = self.opts.model['n_heads'],
            key_size = self.opts.model['key_size'],
            inner_dim = self.opts.model['inner_dim'],
            n_layers = self.opts.model['n_layers'],
            max_len = self.opts.model['dec_max_len'],
            dropout = self.opts.model['dropout'],
            tied_emb_proj = self.opts.model['tied_emb_proj'],
            eps = self.opts.model['label_smoothing'],
            ctx_name = self.sl,
        )

        if self.opts.model['shared_embs']:
            self.enc.embs[0].weight = self.dec.embs[0].weight
    
    def encode(self, batch, **kwargs):
        ctx_dict = {k: [v, None] for k, v in batch.items() if k not in (self.sl, self.tl)}
        
        hs, x, attn = self.enc(
            x=batch[self.sl], 
            z=batch[self.feat_name],
            batch=batch
        )
        feat = batch[self.feat_name]

        ctx_dict[self.sl] = [hs, x]

        return ctx_dict
    
    def forward(self, batch, **kwargs):
        ctx_dict = self.encode(batch)
        y = batch[self.tl]

        result = self.dec(ctx_dict, y)
        result['n_items'] = torch.nonzero(batch[self.tl][1:]).shape[0]

        return result

