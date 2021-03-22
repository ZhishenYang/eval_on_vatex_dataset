import logging

import torch
from torch import nn
import numpy as np
import math

from .transformer import Transformer
from ..layers.transformer import MultiHeadAttention

logger = logging.getLogger('nmtpytorch')

class FusionLayer(nn.Module):
    def __init__(self, model_dim, feat_dim, n_heads, dropout):
        super().__init__()

        self.vis_rec = MultiHeadAttention(
            model_dim = feat_dim,
            n_heads = n_heads,
            dropout = dropout,
            k_dim = model_dim,
            v_dim = model_dim
        )
        self.attn = MultiHeadAttention(
            model_dim=model_dim,
            n_heads = n_heads,
            dropout = dropout,
            k_dim = feat_dim,
            v_dim = feat_dim
        )
        self.layer_norm = nn.LayerNorm(model_dim)
        self.layer_norm_feat = nn.LayerNorm(feat_dim)
    
    def forward(self, h, z):
        # h and z have shape of (t, n_samples, dim).
        # convert (n_samples, t, feat_dim) to operate MHA
        z = z.transpose(0, 1)
        h = h.transpose(0, 1)

        z_dot, _ = self.vis_rec(z, h, h, None)
        z_dot = self.layer_norm_feat(z_dot)

        h_bar, _ = self.attn(h, z_dot, z_dot, None)
        h_bar = self.layer_norm(h_bar)

        #TODO: verify element-wise weighted sum implementation
        h_out = h_bar * h

        # return to nmtpytorch world.
        return h_out.transpose(0, 1)

class VGTShallow(Transformer):
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
        super().setup(is_train)

        self.fusion = FusionLayer(
            model_dim=self.opts.model['model_dim'],
            feat_dim=self.opts.model['feat_dim'],
            n_heads = self.opts.model['n_heads'],
            dropout = self.opts.model['dropout'],
        )
    
    def encode(self, batch, **kwargs):
        ctx_dict = {k: [v, None] for k, v in batch.items() if k not in (self.sl, self.tl)}
        
        hs, x, attn = self.enc(batch[self.sl], batch=batch)
        feat = batch[self.feat_name]

        fused_hs = self.fusion(hs, feat)

        ctx_dict[self.sl] = [fused_hs, x]

        return ctx_dict
    
    def forward(self, batch, **kwargs):
        ctx_dict = self.encode(batch)
        y = batch[self.tl]

        result = self.dec(ctx_dict, y)
        result['n_items'] = torch.nonzero(batch[self.tl][1:]).shape[0]

        return result

