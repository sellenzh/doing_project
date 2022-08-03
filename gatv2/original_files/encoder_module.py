from torch import nn

from attention_module import MultiHeadAttention, GraphMultiHeadAttention
from ffn import FeedForwardNet

class EncoderLayer(nn.Module):

    def __init__(self, inputs, heads, hidden, a_dropout=None, f_dropout=None):
        '''Implemented encoder layer via multi-head self-attention and feedforward net'''
        super(EncoderLayer, self).__init__()
        self.heads = heads
        self.hidden = hidden
        self.inputs = inputs

        self.attention = MultiHeadAttention(heads, inputs, a_dropout=a_dropout, f_dropout=f_dropout)
        self.attention_norm = nn.LayerNorm(inputs)
        self.feedforward = FeedForwardNet(inputs, hidden, dropout=f_dropout)
        self.feedforward_norm = nn.LayerNorm(inputs)

    def forward(self, x, mask=None):
        y = self.attention_norm(x)
        y = self.attention(y, y, y, mask=mask)
        x = x + y
        y = self.feedforward_norm(x)
        y = self.feedforward(y)
        x = x + y
        return x

class GraphEncoderLayer(nn.Module):

    def __init__(self, inputs, heads, hidden, gat_version, a_dropout=None, f_dropout=None, alpha=0.1):
        '''Implemented spatial encoder layer via multi-head spatial self-attention and feedforward net'''
        super(GraphEncoderLayer, self).__init__()
        self.inputs = inputs
        self.heads = heads
        self.hidden = hidden

        self.attention = GraphMultiHeadAttention(heads, inputs, gat_version, a_dropout=a_dropout, f_dropout=f_dropout, alpha=alpha)
        self.attention_norm = nn.LayerNorm(inputs)
        self.feedforward = FeedForwardNet(inputs, hidden, dropout=f_dropout)
        self.feedforward_norm = nn.LayerNorm(inputs)

    def forward(self, h, bias=None):
        y = self.attention_norm(h)
        y = self.attention(y, bias)
        h = h + y

        y = self.feedforward_norm(h)
        y = self.feedforward(y)
        h = h + y
        return h

class Encoder(nn.Module):

    def __init__(self, inputs, heads, hidden, a_dropout=None, f_dropout=None):
        '''Implemented encoder via multiple stacked encoder layers'''
        super(Encoder, self).__init__()
        self.inputs = inputs
        self.heads = heads
        self.hidden = hidden
        self.attention_dropout = a_dropout
        self.feature_dropout = f_dropout

        self.norm = nn.LayerNorm(inputs)
        self.layers = EncoderLayer(inputs, heads, hidden, a_dropout=a_dropout, f_dropout=f_dropout)

    def forward(self, x, mask=None):
        x = self.layers(x, mask)
        return self.norm(x)#x

class GraphEncoder(nn.Module):

    def __init__(self, inputs, heads, hidden, gat_version, a_dropout=None, f_dropout=None, alpha=0.1):
        '''Implemented spatial encoder via multiple stacked spatial encoder layers'''
        super(GraphEncoder, self).__init__()
        self.inputs = inputs
        self.heads = heads
        self.hidden = hidden
        self.attention_dropout = a_dropout
        self.feature_dropout = f_dropout
        self.alpha = alpha

        self.norm = nn.LayerNorm(inputs)
        self.layers = GraphEncoderLayer(inputs, heads, hidden, gat_version, a_dropout=a_dropout, f_dropout=f_dropout, alpha=alpha)
    def forward(self, h, bias=None):
        h = self.layers(h, bias)
        return self.norm(h)
