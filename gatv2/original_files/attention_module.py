import torch
from torch import nn
import math
import torch.nn.functional as F
from entmax import sparsemax, entmax15, entmax_bisect

class ScaledDotProductAttention(nn.Module):

    def __init__(self, dropout=None):
        '''Implemented simple attention'''
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(p=dropout) if dropout is not None else nn.Identity()
        self.attn_type = 'entmax15'

    def forward(self, q, k, v, mask=None):
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e12)
        if self.attn_type == 'softmax':
            attn = F.softmax(scores, dim=-1)
        elif self.attn_type == 'sparsemax':
            attn = sparsemax(scores, dim=-1)
        elif self.attn_type == 'entmax15':
            attn = entmax15(scores, dim=-1)
        elif self.attn_type == 'entmax':
            attn = entmax_bisect(scores, alpha=1.6, dim=-1, n_iter=25)
        return torch.matmul(attn, v)

class MultiHeadAttention(nn.Module):
    def __init__(self, heads, inputs, a_dropout=None, f_dropout=None):
        '''Implemented simple multi-head attention'''
        super(MultiHeadAttention, self).__init__()
        self.heads = heads
        self.inputs = inputs
        assert inputs % heads == 0
        self.hidden = inputs // heads

        self.attention = ScaledDotProductAttention(a_dropout)
        self.linear_q = nn.Linear(inputs, inputs)
        self.linear_k = nn.Linear(inputs, inputs)
        self.linear_v = nn.Linear(inputs, inputs)
        self.output = nn.Linear(inputs, inputs)
        self.dropout = nn.Dropout(p=f_dropout) if f_dropout is not None else nn.Identity()

    def forward(self, q, k, v, mask):
        bs = q.size(0)
        q = self.linear_q(q).view(bs, -1, self.heads, self.hidden).transpose(1, 2)
        k = self.linear_k(k).view(bs, -1, self.heads, self.hidden).transpose(1, 2)
        v = self.linear_v(v).view(bs, -1, self.heads, self.hidden).transpose(1, 2)

        out = self.attention(q, k, v, mask).transpose(1, 2).contiguous()
        out = out.view(bs, -1, self.inputs)
        return self.dropout(self.output(out))



class GraphAttention(nn.Module):
    # modified on pyGAT(https://github.com/Diego999/pyGAT/)
    def __init__(self, inp, outp, gat_version, dropout=None, alpha=0.1, concat=True):
        '''Implemented spatial attention via Graph Attention Layer (GAT)'''
        super(GraphAttention, self).__init__()
        self.inp = inp
        self.outp = outp
        self.alpha = alpha
        self.concat = concat
        self.gat = gat_version
        if self.gat == 'v1':
            self.W = nn.Linear(inp, outp, bias=False)
            self.A_i = nn.Linear(outp, 1, bias=False)
        else:
            self.W = nn.Linear(outp * 2, outp, bias=False)
            self.A_i = nn.Linear(outp , 1, bias=False)
            self.v2 = nn.Linear(inp, outp, bias=False)

        self.A_j = nn.Linear(outp, 1, bias=False)
        self.dropout = nn.Dropout(p=dropout) if dropout is not None else nn.Identity()
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, bias=None):
        # h.shape: (B, N, inp), Wh.shape: (B, N, outp)
        Wh, attention = self._prepare_attentional_mechanism_input(h)
        if not (bias is None):
            attention = attention + bias
        #attention = e.masked_fill(adj == 0, -1e12)
        attention = self.dropout(F.softmax(attention, dim=-1))
        if self.gat == 'v2':
            h_prime = torch.matmul(attention.unsqueeze(2), Wh).squeeze()
            #h_prime = self.v2(h_prime)
        else:
            h_prime = torch.matmul(attention, Wh)
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, h):
        if self.gat == 'v1':
            Wh = self.W(h)
            Wh1 = self.A_i(Wh)
            Wh2 = self.A_j(Wh)
            e = Wh1 + Wh2.transpose(-1, -2)
            return Wh, self.leakyrelu(e)
        else:
            h = self.v2(h)
            hi = h.unsqueeze(2).repeat(1, 1, h.size(1), 1)
            hj = h.unsqueeze(1).repeat(1, h.size(1), 1, 1)
            hij = torch.cat((hi, hj), dim=-1)
            Wh = self.W(hij)
            Wh = self.leakyrelu(Wh)
            Wh1 = self.A_i(Wh).squeeze()
            return Wh, Wh1

class GraphMultiHeadAttention(nn.Module):

    def __init__(self, heads, inputs, gat_version, a_dropout=None, f_dropout=None, alpha=0.1):
        '''Implemented multi-head spatial attention via multiple stacked Graph Attention Layers (GATs)'''
        super(GraphMultiHeadAttention, self).__init__()
        self.heads = heads
        self.inputs = inputs
        assert inputs % heads == 0
        self.hidden = inputs // heads
        self.alpha = alpha

        attentions = [GraphAttention(inputs, self.hidden, gat_version, a_dropout, alpha, concat=True) for _ in range(heads)]
        self.attentions = nn.ModuleList(attentions)
        self.output = nn.Linear(inputs, inputs)
        self.dropout = nn.Dropout(p=f_dropout) if f_dropout is not None else nn.Identity()

    def forward(self, h, bias=None):
        if not (bias is None):
            biases = bias.chunk(self.heads, -1)  # 8 x [124, 19, 19, 1]
            out = torch.cat([attention(h, b.squeeze(-1)) for attention, b in zip(self.attentions, biases)], dim=-1)
        else:
            out = torch.cat([attention(h, None) for attention in self.attentions], dim=-1)
        return self.dropout(F.elu(self.output(out)))
