import torch
from torch import nn
from torch.autograd import Variable

import math, random

class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature, dropout):
        super(ScaledDotProductAttention, self).__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, Q, K, V, attn_mask):
        '''
        Q, K, V: (batch_size x n_heads) x len x d_k(d_v)
        '''

        attn = torch.bmm(Q, K.transpose(-1, -2))
        attn = attn / self.temperature
        
        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask, -math.inf)
        
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        
        context = torch.bmm(attn, V)

        return context, attn

class MultiHeadAttention(nn.Module):
    def __init__(self, model_dim, n_heads, dropout, k_dim=None, v_dim=None):
        super(MultiHeadAttention, self).__init__()
        assert model_dim % n_heads == 0

        self.model_dim = model_dim
        self.n_heads = n_heads
        self.d_k = model_dim // n_heads

        self.k_dim = k_dim if k_dim else model_dim
        self.v_dim = v_dim if v_dim else model_dim

        self.W_Q = nn.Linear(model_dim, model_dim)
        self.W_K = nn.Linear(self.k_dim, model_dim)
        self.W_V = nn.Linear(self.v_dim, model_dim)
        self.W_O = nn.Linear(model_dim, model_dim)

        self.scaled_dot_product = ScaledDotProductAttention(math.sqrt(self.d_k), dropout)

        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V, mask=None):
        '''
        Q, K, V: batch_size x len x model_dim
        '''
        batch_size = Q.size(0)
        residual = Q

        def pack(x):
            # x: batch_size x len x model_dim
            # => batch_size x len x n_heads x d_k
            x = x.view(batch_size, -1, self.n_heads, self.d_k)
            # => batch_size x n_heads x len x d_k
            x = x.transpose(1, 2).contiguous()
            # => (batch_size x n_heads) x len x d_k
            x = x.view(batch_size * self.n_heads, -1, self.d_k)

            return x
        
        def unpack(x):
            # x: (batch_size x n_heads) x len x d_k
            # => batch_size x n_heads x len x d_k
            x = x.view(batch_size, self.n_heads, -1, self.d_k)
            # => batch_size x len x n_heads x d_k
            x = x.transpose(1, 2).contiguous()
            # => batch_size x len x model_dim
            x = x.view(batch_size, -1, self.model_dim)

            return x

        Q = pack(self.W_Q(Q))
        K = pack(self.W_K(K))
        V = pack(self.W_V(V))
        # batch_size x len x len 
        # => (batch_size x n_heads) x len x len
        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)
            mask = mask.view(-1, mask.size(2), mask.size(3))
        
        heads, attns = self.scaled_dot_product(Q, K, V, mask)
        heads = unpack(heads)

        outputs = self.W_O(heads)
        outputs = self.dropout(outputs)

        return outputs, attns

class PositionwiseFeedForward(nn.Module):
    def __init__(self, model_dim, inner_dim, dropout):
        super(PositionwiseFeedForward, self).__init__()
        self.model_dim = model_dim
        self.inner_dim = inner_dim

        self.w_1 = nn.Linear(model_dim, inner_dim)
        self.w_2 = nn.Linear(inner_dim, model_dim)

        self.relu = nn.ReLU() # max(0, ...)
        self.do = nn.Dropout(dropout)

    def forward(self, x):
        '''
        x: batch_size x len x model_dim
        '''
        residual = x

        outputs = self.do(self.relu(self.w_1(x)))
        outputs = self.do(self.w_2(outputs))

        return outputs

class PositionalEncoding(nn.Module):
    def __init__(self, model_dim, max_len):
        super(PositionalEncoding, self).__init__()

        pos = torch.arange(0, max_len).float().unsqueeze(1) # max_len x 1
        div = torch.arange(0, model_dim // 2).float()
        div = torch.pow(10000, div * 2 / model_dim) # model_dim

        pe = torch.zeros(max_len, model_dim)
        pe[:, 0::2] = torch.sin(pos / div)
        pe[:, 1::2] = torch.cos(pos / div)
        pe = pe.unsqueeze(0)    # 1 x max_len x model_dim
        self.register_buffer('pe', pe)

    def forward(self, emb):
        pos_emb = self.pe[:, :emb.size(1)] # 1 x len_x x model_dim
        outputs = emb + Variable(pos_emb, requires_grad=False)
        return outputs

class LabelSmoothingLoss(nn.Module):
    def __init__(self, eps, n_vocab, padding_index=0):
        super(LabelSmoothingLoss, self).__init__()

        self.eps = eps
        self.n_vocab = n_vocab
        self.padding_index = padding_index

        unk_eps = eps / (n_vocab - 1)
        one_hot = torch.full((n_vocab,), unk_eps)
        one_hot[padding_index] = 0
        self.register_buffer('one_hot', one_hot.unsqueeze(0))

        self.confidence = 1.0 - eps
        
    def forward(self, pred, y):
        # pred: batch_size x len x n_vocab
        # y: batch_size x len
        pred = pred.view(-1, self.n_vocab)
        y = y.view(-1)

        pad_mask = (y == self.padding_index)

        # smoothed probability over whole vocabulary
        gold = self.one_hot.repeat(y.size(0), 1)
        gold.scatter_(1, y.unsqueeze(-1), self.confidence)
        gold.masked_fill_(pad_mask.unsqueeze(-1).bool(), 0)

        return nn.functional.kl_div(pred, gold, reduction='sum')

def get_pad_attn_mask(q, k, masking_seq=False, padding_idx=0):
    '''
    create attention mask matrix (batch_size x len x len) 
    from input data (batch_size x len x model_dim).
    '''
    n_batch, len_q = q.size()
    n_batch, len_k = k.size()
    # set True for padding cell
    pad_attn_mask = k.data.eq(padding_idx)
    # transform single mask vector (batch_size x len) to
    # mask matrix (batch_size x len x len) by repeating the vector
    pad_attn_mask = pad_attn_mask.unsqueeze(1).expand(n_batch, len_q, len_k)

    if masking_seq == True:
        seq_mask = torch.ones(len_q, len_k, device=pad_attn_mask.device, dtype=torch.uint8)
        seq_mask = torch.triu(seq_mask, diagonal=1).bool()
        pad_attn_mask = pad_attn_mask | seq_mask

    return pad_attn_mask

def get_pad_mask(x, padding_idx=0):
    '''
    x: batch_size x len
    '''
    # assuming used by output.masked_fill(ret, 0)
    # output: batch_size x len x model_dim
    # ret   : batch_size x len x 1
    return x.eq(padding_idx).unsqueeze(-1)

def get_embs(n_vocab, max_len, model_dim, dropout):
    embs = [
        nn.Embedding(n_vocab, model_dim, padding_idx=0),
        nn.Tanh(),
        PositionalEncoding(model_dim, max_len),
        nn.LayerNorm(model_dim),
        nn.Dropout(dropout),
    ]

    return nn.Sequential(*embs)

class FusedMultiHeadAttention(nn.Module):
    def __init__(self, model_dim, feat_dim, feat_ratio, n_heads, dropout, dropnet):
        super(FusedMultiHeadAttention, self).__init__()
        assert model_dim % n_heads == 0
        assert (0 < feat_ratio and feat_ratio < 1) or (dropnet is None), \
            "set dropnet to None when use feat_ratio of 0 or 1."

        self.ctx_mha  = MultiHeadAttention(model_dim, n_heads, dropout)
        self.feat_mha = MultiHeadAttention(model_dim, n_heads, dropout, k_dim=feat_dim, v_dim=feat_dim)

        self.dropout = nn.Dropout(dropout)

        self.feat_ratio = feat_ratio

        # distribute dropnet probability to textual or visual modalities
        self.dropnet = dropnet / 2 if dropnet is not None else dropnet
    
    def forward(self, Q, K, V, mask, feat_K, feat_V, feat_mask):
        residual = Q

        frand = random.random()
        if (not self.training) or \
           (self.dropnet is None) or \
           (self.dropnet < frand and frand < (1 - self.dropnet)):
            ctx_outputs,  ctx_attns  = self.ctx_mha(Q, K, V, mask)
            feat_outputs, feat_attns = self.feat_mha(Q, feat_K, feat_V, feat_mask)

            outputs = (1 - self.feat_ratio) * ctx_outputs + self.feat_ratio * feat_outputs
            
            attns = [ctx_attns, feat_attns]

        elif frand < self.dropnet:
            outputs,  ctx_attns  = self.ctx_mha(Q, K, V, mask)
            
            attns = [ctx_attns, None]

        elif frand > (1 - self.dropnet):
            outputs, feat_attns = self.feat_mha(Q, feat_K, feat_V, feat_mask)

            attns = [None, feat_attns]
            
        else:
            raise Exception("Something wronge with 'dropnet' option.")

        return outputs, attns