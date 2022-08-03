import torch
from torch import nn
import math
from math import sqrt
from torch.autograd import Variable

from model_0730.encoder_module import Encoder, GraphEncoder


class Pose_Module(nn.Module):
    def __init__(self, config, first):
        super(Pose_Module, self).__init__()
        self.in_ch = 4 if first else 32
        self.out_ch = 32 if first else 64
        self.feature_dims = 64 if first else 128 #config.attention_dims
        self.embedding_hiddens = 256 ##@@
        self.num_heads = 8#config.num_heads # 256
        self.hidden_dims = 256#config.ffn_dims
        self.attention_dropout = 0.2#config.attention_dropout if 1. > config.attention_dropout > 0. else None
        self.num_times = 2#config.num_times
        self.gat_version = 'v2'#config.gat_version

        self.posi_en = Posi_Enconding(self.feature_dims, 0.0)
        self.poselinear = nn.Linear(self.in_ch, self.feature_dims, bias=False)
        self.embedding_layer = nn.ModuleList(
            nn.Sequential(nn.Linear(self.in_ch, self.hidden_dims, bias=False), nn.PReLU()) for _ in range(self.num_times))
        self.embed = nn.ModuleList(
            Embedding_module(self.hidden_dims, self.num_heads) for _ in range(self.num_times))
        self.gat_en = nn.ModuleList(
            GraphEncoder(self.feature_dims, self.num_heads, self.hidden_dims, self.gat_version, self.attention_dropout) for _ in range(self.num_times))
        self.gat_de = nn.ModuleList(
            nn.Sequential(nn.Linear(self.feature_dims, self.feature_dims)) for _ in range(self.num_times))
        
        self.tat_en = nn.ModuleList(
            Encoder(self.feature_dims, self.num_heads, self.hidden_dims, self.attention_dropout) for _ in range(self.num_times))
        #self.temporal_token = nn.Parameter(torch.empty(1, 1, self.feature_dims))
        self.output = nn.Linear(self.feature_dims, self.out_ch, bias=False)
        self.activation = nn.LeakyReLU()

        for module in self.modules():
            if isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            '''elif isinstance(module, nn.Embedding):
                nn.init(module.weight)'''
        #nn.init.normal_(self.temporal_token, 0.0, 0.02)
        #nn.init.normal_(self.embedding_layer, 0.0, 0.02)
    
    def forward(self, pose):
        """input: pose -> [2, 4/32, 62, 19]
        output: y -> [2, 32/64, 62, 19]"""
        B, C, T, V = pose.size()
        pose = pose.permute(0, 2, 3, 1).contiguous().view(B, T, V, -1) #[2, 4, 62, 19] -> [2, 62, 19, 4]
        x = self.poselinear(pose) #[2, 62, 19, 4] -> [2, 62, 19, 64/128]
        y = x.contiguous().view(B*T, V, -1)
        bias_pre = pose.unsqueeze(2) - pose.unsqueeze(3) # [2, 62, 19, 19, 4]
        for i in range(self.num_times):
            bias = self.embed[i](
                self.embedding_layer[i](bias_pre)).contiguous().view(
                    -1, V, V, self.num_heads)#[2, 62, 19, 4] -> [2, 62, 19, 8] -> [124, 19, 19, 8]
            gat_out = self.gat_en[i](y, bias)
            gat_out = self.gat_de[i](gat_out)#[124, 19, 64]
            y = gat_out.contiguous().view(B*V, T, -1)#[38, 62, 64/128]
            y = self.posi_en(y) # -> [38, 62, 64/128]
        #for i in range(self.num_times):
            memory = y
            y = self.tat_en[i](y)
            y = (y + memory).contiguous().view(B*T, V, -1)#[B*V, T, C] = [38, 62, 64] -> [B*T, V, C]
        y = self.activation(self.output(y))
        return y.contiguous().view(B, -1, T, V)#[2, 32/64, 62, 19]


class Embedding_module(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False):
        super(Embedding_module, self).__init__()
        self.out_ch = out_channels
        self.linear = nn.Linear(in_channels, out_channels, bias=bias)
    
    def forward(self, x):
        return self.linear(x) / sqrt(self.out_ch)

class Posi_Enconding(nn.Module):
    def __init__(self, dim, dropout, len=62) -> None:
        super(Posi_Enconding, self).__init__()
        self.dropout = nn.Dropout(dropout) if not (dropout is None) else nn.Identity()

        pe = torch.zeros(len, dim)
        position = torch.arange(0, len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) *
                            -(math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        shape = x.size()
        x = x.view(-1, x.size(-2), x.size(-1))
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        x = self.dropout(x)
        x = x.view(*shape)
        return x


config = None
model1 = Pose_Module(config, True)
model2 = Pose_Module(config, False)
#model_embed = Embedding_module(256, 64)
t = torch.randn(size=(2, 4, 62, 19))
#layers = nn.Sequential(nn.Linear(4, 256), nn.PReLU())
#y = layers(t)
#y = model_embed(y)
y = model1(t)
y = model2(y)
print(y.shape)