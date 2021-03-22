import logging

import torch
from torch import nn
import numpy as np
import math

from ..layers import TransformerEncoder, FusionTransformerDecoder
from .transformer import Transformer
from ..utils.misc import get_n_params
from ..utils.data import sort_batch
from ..vocabulary import Vocabulary
from ..utils.topology import Topology
from ..utils.ml_metrics import Loss
from ..utils.device import DEVICE
from ..utils.misc import pbar
from ..datasets import MultimodalDataset
from ..metrics import Metric

logger = logging.getLogger('nmtpytorch')

class MultimodalTransformer(Transformer):
    supports_beam_search = True

    def set_defaults(self):
        super().set_defaults()
        self.defaults.update({
            'dec_layer_types': 'ffffff',
            'feat_ratio': 0.5,
            'dropnet': 0.0,
            'feat_name': 'feats',
            'feat_dim': 2048,
            'n_feats': 10,
        })

    def __init__(self, opts):
        super().__init__(opts)

        self.feat_name = self.opts.model['feat_name']

    def reset_parameters(self):
        super().reset_parameters()

    def setup(self, is_train=True):
        self.enc = TransformerEncoder(
            n_vocab = self.n_src_vocab,
            model_dim = self.opts.model['model_dim'],
            n_heads = self.opts.model['n_heads'],
            key_size = self.opts.model['key_size'],
            inner_dim = self.opts.model['inner_dim'],
            n_layers = self.opts.model['n_layers'],
            max_len = self.opts.model['max_len'],
            dropout = self.opts.model['dropout'],
        )
        self.dec = FusionTransformerDecoder(
            n_vocab = self.n_trg_vocab,
            model_dim = self.opts.model['model_dim'],
            n_heads = self.opts.model['n_heads'],
            key_size = self.opts.model['key_size'],
            inner_dim = self.opts.model['inner_dim'],
            n_layers = self.opts.model['n_layers'],
            max_len = self.opts.model['max_len'],
            dropout = self.opts.model['dropout'],
            tied_emb_proj = self.opts.model['tied_emb_proj'],
            eps = self.opts.model['label_smoothing'],
            ctx_name = self.sl,
            layer_types = self.opts.model['dec_layer_types'],
            feat_name = self.feat_name,
            feat_dim = self.opts.model['feat_dim'],
            feat_ratio = self.opts.model['feat_ratio'],
            dropnet = self.opts.model['dropnet'],
        )

        if self.opts.model['shared_embs']:
            self.enc.embs[0].weight = self.dec.embs[0].weight
    
    def encode(self, batch, **kwargs):
        ctx_dict = super().encode(batch, **kwargs)
        ctx_dict[self.feat_name] = (batch[self.feat_name], None)

        return ctx_dict
    
    def forward(self, batch, **kwargs):
        ctx_dict = self.encode(batch)
        y = batch[self.tl]

        result = self.dec(ctx_dict, y)
        result['n_items'] = torch.nonzero(batch[self.tl][1:]).shape[0]

        return result
