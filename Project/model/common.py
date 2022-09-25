import math
import torch
from torch import nn
import numpy as np

import torch.nn.functional as F
from entmax import entmax15

def conv_init(conv):
    if conv.weight is not None:
        nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)

def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


class DataBN(nn.Module):
    def __init__(self, dims, args):
        super(DataBN, self).__init__()
        self.dims = dims
        self.nodes = args.nodes

        self.bn = nn.BatchNorm1d(self.dims * self.nodes)
        bn_init(self.bn, 1)
        self.drop = nn.Dropout(args.dropout_rate)

    def forward(self, pose):
        N, C, T, V = pose.shape
        pose = pose.permute(0, 1, 3, 2).contiguous().view(N, C * V, T)
        pose = self.bn(pose)
        y = pose.view(N, C, V, T).permute(0, 1, 3, 2).contiguous()
        return y


class Conv2BN2Silu(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=1,
                 stride=1, padding=0, bias=True):
        unit = [nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                          kernel_size=kernel_size, stride=stride,
                          padding=padding, bias=bias)]
        unit.append(nn.BatchNorm2d(out_channels))
        unit.append(nn.SiLU())
        super(Conv2BN2Silu, self).__init__(*unit)


class Conv1BN1Silu(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=1,
                 stride=1, padding=0, bias=True):
        unit = [nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                          kernel_size=kernel_size, stride=stride,
                          padding=padding, bias=bias)]
        unit.append(nn.BatchNorm1d(out_channels))
        unit.append(nn.SiLU())
        super(Conv1BN1Silu, self).__init__(*unit)


class ImgConvLayers(nn.Module):
    def __init__(self, ch, ch1, ch2):
        super(ImgConvLayers, self).__init__()
        self.ch, self.ch1, self.ch2 = ch, ch1, ch2

        self.img1 = Conv2BN2Silu(
            in_channels=self.ch, out_channels=self.ch1,
            kernel_size=3, stride=1, padding=0, bias=False
        )
        self.img2 = Conv2BN2Silu(
            in_channels=self.ch1, out_channels=self.ch1,
            kernel_size=3, stride=1, padding=0, bias=False
        )
        self.img3 = Conv2BN2Silu(
            in_channels=self.ch1, out_channels=self.ch2,
            kernel_size=2, stride=1, padding=0, bias=False
        )

    def forward(self, img):
        y = self.img3(self.img2(self.img1(img)))
        return y


class VelConvLayers(nn.Module):
    def __init__(self, ch, ch1, ch2):
        super(VelConvLayers, self).__init__()
        self.ch, self.ch1, self.ch2 = ch, ch1, ch2

        self.vel1 = Conv1BN1Silu(
            in_channels=2, out_channels=self.ch1,
            kernel_size=3, stride=1, padding=0, bias=False
        )
        self.vel2 = Conv1BN1Silu(
            in_channels=self.ch1, out_channels=self.ch1,
            kernel_size=3, stride=1, padding=0, bias=False
        )
        self.vel3 = Conv1BN1Silu(
            in_channels=self.ch1, out_channels=self.ch2,
            kernel_size=2, stride=1, padding=0, bias=False
        )
    def forward(self, vel):
        y1 = self.vel2(self.vel1(vel))
        y2 = self.vel3(y1)
        return y1, y2


class Att(nn.Sequential):
    def __init__(self, dims):
        layers = [nn.SiLU()]
        layers.append(nn.Linear(dims, dims, bias=False))
        layers.append(nn.BatchNorm1d(dims))
        layers.append(nn.Sigmoid())
        super(Att, self).__init__(*layers)


class Process(nn.Module):
    def __init__(self, dims, n_clss):
        super(Process, self).__init__()

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.att = Att(dims)

        self.linear = nn.Linear(dims, n_clss)
        nn.init.normal_(self.linear.weight, 0, math.sqrt(2. / n_clss))

        self.drop = nn.Dropout(0.3)

    def forward(self, x):
        x = self.gap(x).squeeze(-1)
        x = x.squeeze(-1)
        y = self.att(x).mul(x) + x
        y = self.linear(self.drop(y))

        return y


class GCN_TAT_layers(nn.Module):
    def __init__(self, ch, ch1, ch2, args):
        super(GCN_TAT_layers, self).__init__()
        self.layer1 = GCN_TAT_unit(ch, ch1, args, residual=False)
        self.layer2 = GCN_TAT_unit(ch1, ch2, args)

        self.pool_sig_1d = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Sigmoid()
        )
        self.pool_sig_2d = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Sigmoid()
        )

    def forward(self, pose, img, vel1, vel2):
        y = self.layer1(pose)
        y = y.mul(self.pool_sig_1d(vel1).unsqueeze(-1))
        y = self.layer2(y)
        y = y.mul(self.pool_sig_2d(img))
        y = y.mul(self.pool_sig_1d(vel2).unsqueeze(-1))
        return y


class GCN_TAT_unit(nn.Module):
    def __init__(self, in_channels, out_channels, args,
                 stride=1, residual=True, adaptive=True):
        super(GCN_TAT_unit, self).__init__()
        self.feature_dims = out_channels * 3
        self.head_num = args.head_num
        self.hidden_dims = out_channels * 6
        self.rate = args.dropout_rate
        self.tat_num = args.layer_num
        self.res = residual

        self.gcn = DecouplingGCN(in_channels, out_channels, args, adaptive=True)
        self.embed = Embedding(out_channels, self.feature_dims)

        self.time_att = nn.ModuleList(
            Encoder(inputs=self.feature_dims, hidden=self.hidden_dims, args=args)
            for _ in range(self.tat_num))
        self.relu = nn.ReLU()

        if not self.res:
            self.residual = lambda x: 0
        elif in_channels == out_channels and stride == 1:
            self.residual = lambda x: x
        else:
            self.residual = ConvResidual(in_channels, out_channels, kernel_size=1, stride=stride)

        self.linear = nn.Linear(self.feature_dims, out_channels)

    def forward(self, x):
        gcn = self.gcn(x)
        res = self.residual(x)
        x = gcn + res

        B, C, T, V = x.size()
        tcn = self.embed(x.permute(0, 3, 2, 1)).contiguous().view(B * V, T, -1)
        for i in range(self.tat_num):
            memory = tcn
            tcn = self.time_att[i](tcn)
            tcn += memory
        y = self.linear(tcn).contiguous().view(B, -1, T, V)
        return self.relu(y)


class ConvResidual(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1):
        super(ConvResidual, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        y = self.conv(x)
        return self.relu(self.bn(y))


class DecouplingGCN(nn.Module):
    def __init__(self, in_channels, out_channels,
                 args, adaptive=True):
        super(DecouplingGCN, self).__init__()
        self.in_ch = in_channels
        self.out_ch = out_channels
        self.groups = args.groups
        self.nodes = args.nodes
        self.device = args.device
        self.adaptive = adaptive
        self.A = np.stack([np.eye(self.nodes)] * 3, axis=0)
        self.subnet = self.A.shape[0]

        self.linear = nn.Linear(self.in_ch, self.out_ch)
        if self.adaptive:
            self.DecoupleA = nn.Parameter(torch.tensor(
                np.reshape(self.A.astype(np.float32), [3, 1, 19, 19]), dtype=torch.float32,
                requires_grad=True).repeat(1, self.groups, 1, 1), requires_grad=True)
        else:
            self.DecoupleA = torch.autograd.Variable(torch.from_numpy(self.A.astype(np.float32)), requires_grad=False)

        if self.in_ch != self.out_ch:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x

        self.bn0 = nn.BatchNorm2d(self.out_ch * self.subnet)
        self.bn1 = nn.BatchNorm2d(self.out_ch)
        self.relu = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn1, 1e-6)

        self.Linear_weight = nn.Parameter(torch.zeros(
            self.in_ch, self.out_ch * self.subnet, requires_grad=True, device=self.device), requires_grad=True)
        nn.init.normal_(self.Linear_weight, 0, math.sqrt(
            0.5 / (self.out_ch * self.subnet)))

        self.Linear_bias = nn.Parameter(torch.zeros(
            1, self.out_ch * self.subnet, 1, 1, requires_grad=True, device=self.device), requires_grad=True)
        nn.init.constant_(self.Linear_bias, 1e-6)

    def L2_norm(self, A):
        A_norm = torch.norm(A, 2, dim=1, keepdim=True) + 1e-4
        A = A / A_norm
        return A

    def forward(self, x):

        if self.adaptive:
            learn_A = self.DecoupleA.repeat(1, self.out_ch // self.groups, 1, 1)
            norm_learn_A = torch.cat([self.L2_norm(learn_A[0:1, ...]),
                                      self.L2_norm(learn_A[1:2, ...]),
                                      self.L2_norm(learn_A[2:3, ...])], 0)
        else:
            norm_learn_A = self.DecoupleA.cuda(x.get_device()).unsqueeze(1)

        y = torch.einsum('nctv,cd->ndtv', (x, self.Linear_weight)).contiguous()
        y += self.Linear_bias
        y = self.bn0(y)

        n, kc, t, v = y.size()
        y = y.view(n, self.subnet, kc // self.subnet, t, v)
        y = torch.einsum('nkctv,kcvw->nctw', (y, norm_learn_A))
        y = self.bn1(y)
        y += self.down(x)
        return self.relu(y)


class Embedding(nn.Module):
    def __init__(self, in_channel, out_channels, bias=False):
        super(Embedding, self).__init__()
        self.out_ch = out_channels
        self.linear = nn.Linear(in_channel, out_channels, bias=bias)

    def forward(self, x):
        return self.linear(x) / math.sqrt(self.out_ch)


class Encoder(nn.Module):
    def __init__(self, inputs, hidden, args):
        super(Encoder, self).__init__()

        self.norm = nn.LayerNorm(inputs)
        self.layers = EncoderLayers(inputs, hidden, args)

    def forward(self, x):
        y = self.layers(x)
        return self.norm(y)


class EncoderLayers(nn.Module):
    def __init__(self, inputs, hidden, args):
        super(EncoderLayers, self).__init__()
        self.inputs = inputs
        self.hidden = hidden
        self.rate = args.dropout_rate
        self.heads = args.head_num

        self.attention = MultiHeadAttention(self.heads, self.inputs, self.rate)
        self.attention_norm = nn.LayerNorm(self.inputs)

        self.feedforward = FeedForwardNet(self.inputs, self.hidden, self.rate)
        self.feedforward_norm = nn.LayerNorm(self.inputs)

    def forward(self, x):
        y = self.attention_norm(x)
        y = self.attention(y, y, y)
        x += y
        y = self.feedforward_norm(x)
        y = self.feedforward(y)
        x += y
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, inputs, rate):
        super(MultiHeadAttention, self).__init__()
        self.heads = heads
        self.inputs = inputs
        assert inputs % heads == 0
        self.hidden = inputs // heads
        self.rate = rate

        self.attention = ScaledDotProductAttention(self.rate)
        self.linear_q = nn.Linear(inputs, inputs)
        self.linear_k = nn.Linear(inputs, inputs)
        self.linear_v = nn.Linear(inputs, inputs)

        self.output = nn.Linear(inputs, inputs)
        self.dropout = nn.Dropout(rate) if rate is not None else nn.Identity()

    def forward(self, q, k, v):
        bs = q.size(0)
        q_w = self.linear_q(q).view(bs, -1, self.heads, self.hidden).transpose(1, 2)
        k_w = self.linear_k(k).view(bs, -1, self.heads, self.hidden).transpose(1, 2)
        v_w = self.linear_v(v).view(bs, -1, self.heads, self.hidden).transpose(1, 2)

        out = self.attention(q_w, k_w, v_w).transpose(1, 2).contiguous()
        out = out.view(bs, -1, self.inputs)
        return self.dropout(self.output(out))


class ScaledDotProductAttention(nn.Module):
    def __init__(self, rate=None):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(rate) if rate is not None else nn.Identity()
        self.attn_type = 'entmax15'

    def forward(self, q, k, v):
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
        att = entmax15(scores, dim=-1)
        return torch.matmul(att, v)


class FeedForwardNet(nn.Module):
    def __init__(self, inputs, hidden, rate=None):
        super(FeedForwardNet, self).__init__()
        self.upscale = nn.Linear(inputs, hidden)
        self.activation = Mish()
        self.downscale = nn.Linear(hidden, inputs)
        self.dropout = nn.Dropout(rate) if rate is not None else nn.Identity()

    def forward(self, x):
        y = self.upscale(x)
        y = self.activation(y)
        y = self.downscale(y)
        return self.dropout(y)


class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        return x * (torch.tanh(F.softplus(x)))
