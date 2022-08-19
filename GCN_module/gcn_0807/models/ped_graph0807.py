
import math
import torch
from torch import nn
import numpy as np
from math import sqrt

from model_0730.encoder_module import Encoder
    

class pedMondel(nn.Module):

    def __init__(self, frames, vel=False, seg=False, h3d=True, nodes=19, n_clss=1):
        super(pedMondel, self).__init__()

        self.h3d = h3d # bool if true 3D human keypoints data is enable otherwise 2D is only used
        self.frames = frames
        self.vel = vel
        self.seg = seg
        self.n_clss = n_clss
        self.ch = 4 if h3d else 3
        self.ch1, self.ch2 = 32, 64
        i_ch = 4 if seg else 3

        self.data_bn = nn.BatchNorm1d(self.ch * nodes)
        bn_init(self.data_bn, 1)
        self.drop = nn.Dropout(0.25)
        A = np.stack([np.eye(nodes)] * 3, axis=0)
        B = np.stack([np.eye(nodes)] * 3, axis=0)
        
        if frames:
            self.conv0 = nn.Sequential(
                nn.Conv2d(i_ch, self.ch1, kernel_size=3, stride=1, padding=0, bias=False), 
                nn.BatchNorm2d(self.ch1), nn.SiLU())
        if vel:
            self.v0 = nn.Sequential(
                nn.Conv1d(2, self.ch1, 3, bias=False), nn.BatchNorm1d(self.ch1), nn.SiLU())
        # ----------------------------------------------------------------------------------------------------
        self.l1 = TCN_GCN_unit(self.ch, self.ch1, A, residual=False)

        if frames:
            self.conv1 = nn.Sequential(
                nn.Conv2d(self.ch1, self.ch1, kernel_size=3, stride=1, padding=0, bias=False), 
                nn.BatchNorm2d(self.ch1), nn.SiLU())
        if vel:
            self.v1 = nn.Sequential(
                nn.Conv1d(self.ch1, self.ch1, 3, bias=False), 
                nn.BatchNorm1d(self.ch1), nn.SiLU())
        # ----------------------------------------------------------------------------------------------------
        self.l2 = TCN_GCN_unit(self.ch1, self.ch2, A)

        if frames:
            self.conv2 = nn.Sequential(
                nn.Conv2d(self.ch1, self.ch2, kernel_size=2, stride=1, padding=0, bias=False), 
                nn.BatchNorm2d(self.ch2), nn.SiLU())
            
        if vel:
            self.v2 = nn.Sequential(
                nn.Conv1d(self.ch1, self.ch2, kernel_size=2, bias=False), 
                nn.BatchNorm1d(self.ch2), nn.SiLU())

        self.gap = nn.AdaptiveAvgPool2d(1)
        
        self.att = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.ch2, self.ch2, bias=False),
            nn.BatchNorm1d(self.ch2), 
            nn.Sigmoid()
        )

        self.linear = nn.Linear(self.ch2, self.n_clss)
        nn.init.normal_(self.linear.weight, 0, math.sqrt(2. / self.n_clss))
        # pooling sigmoid fucntion for image feature fusion
        self.pool_sigm_2d = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Sigmoid()
        )
        if vel:
            self.pool_sigm_1d = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Sigmoid()
            )
        
    
    def forward(self, kp, frame=None, vel=None): 

        N, C, T, V = kp.shape
        kp = kp.permute(0, 1, 3, 2).contiguous().view(N, C * V, T)
        kp = self.data_bn(kp)
        kp = kp.view(N, C, V, T).permute(0, 1, 3, 2).contiguous()#[2, 4, T, 19]
        
        if self.frames:
            f1 = self.conv0(frame) #[2, 32, 190, 62]
        if self.vel:
            v1 = self.v0(vel)#[2, 32, T-2]

        x1 = self.l1(kp)
        if self.frames:
            f1 = self.conv1(f1)   
            x1.mul(self.pool_sigm_2d(f1))
        if self.vel:   
            v1 = self.v1(v1)
            x1 = x1.mul(self.pool_sigm_1d(v1).unsqueeze(-1))

        x1 = self.l2(x1)
        if self.frames:
            f1 = self.conv2(f1) 
            x1 = x1.mul(self.pool_sigm_2d(f1))
        if self.vel:  
            v1 = self.v2(v1)
            x1 = x1.mul(self.pool_sigm_1d(v1).unsqueeze(-1))

        x1 = self.gap(x1).squeeze(-1)
        x1 = x1.squeeze(-1)
        x1 = self.att(x1).mul(x1) + x1
        x1 = self.drop(x1)
        x1 = self.linear(x1)

        return x1


def conv_init(conv):
    if conv.weight is not None:
        nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


def conv_branch_init(conv, branches):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(weight, 0, math.sqrt(2. / (n * k1 * k2 * branches)))
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)


class unit_tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1):
        super(unit_tcn, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1))

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x

class decoupling_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, A, adaptive) -> None:
        super(decoupling_gcn, self).__init__()
        self.in_ch = in_channels
        self.out_ch = out_channels
        self.subnet = A.shape[0]
        self.groups = 8
        self.adaptive = adaptive

        self.linear = nn.Linear(self.in_ch, self.out_ch)

        self.DecoupleA = nn.Parameter(torch.tensor(np.reshape(A.astype(np.float32), [
                                      3, 1, 19, 19]), dtype=torch.float32, requires_grad=True).repeat(1, self.groups, 1, 1), requires_grad=True)

        if adaptive:
            self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)), requires_grad=True)
        else:
            self.A = torch.autograd.Variable(torch.from_numpy(A.astype(np.float32)), requires_grad=False)

        
        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x

        self.bn0 = nn.BatchNorm2d(out_channels * self.subnet)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)

        self.Linear_weight = nn.Parameter(torch.zeros(
            in_channels, out_channels * self.subnet, requires_grad=True, device='cuda'), requires_grad=True)
        nn.init.normal_(self.Linear_weight, 0, math.sqrt(
            0.5 / (out_channels * self.subnet)))

        self.Linear_bias = nn.Parameter(torch.zeros(
            1, out_channels * self.subnet, 1, 1, requires_grad=True, device='cuda'), requires_grad=True)
        nn.init.constant(self.Linear_bias, 1e-6)

    def L2_norm(self, A):
        # A:N,V,V
        A_norm = torch.norm(A, 2, dim=1, keepdim=True) + 1e-4  # N,1,V
        A = A / A_norm
        return A

    def forward(self, x):
        learn_A = self.DecoupleA.repeat(1, self.out_ch // self.groups, 1, 1)# learn_A -> [3, 32, 19, 19]
        norm_learn_A = torch.cat([self.L2_norm(learn_A[0:1, ...]),
                            self.L2_norm(learn_A[1:2, ...]),
                            self.L2_norm(learn_A[2:3, ...])],
                            0)
        #norm_A -> [3, 32, 19, 19]
        y = torch.einsum('nctw,cd->ndtw', (x, self.Linear_weight)).contiguous()
        y = y + self.Linear_bias
        y = self.bn0(y)

        n, kc, t, v = y.size()
        y = y.view(n, self.subnet, kc // self.subnet, t, v)
        y = torch.einsum('nkctv,kcvw->nctw', (y, norm_learn_A))

        y = self.bn(y)
        y += self.down(x)
        y = self.relu(y)
        return y

class TCN_GCN_unit(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True, adaptive=True):
        super(TCN_GCN_unit, self).__init__()
        self.feature_dims = 128
        self.num_heads = 8
        self.hidden_dims = 256
        self.attention_dropout = 0.2
        self.num_times = 5

        #self.gcn1 = unit_gcn(in_channels, out_channels, A, adaptive=adaptive)
        self.gcn1 = decoupling_gcn(in_channels, out_channels, A, adaptive)
        self.embed = Embedding_module(out_channels, self.feature_dims)
        #self.tcn1 = unit_tcn(out_channels, out_channels, stride=stride)
        self.tcn1 = nn.ModuleList(
            Encoder(self.feature_dims, self.num_heads, self.hidden_dims, self.attention_dropout) for _ in range(self.num_times))
        self.relu = nn.ReLU(inplace=True)
        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)
        self.linear = nn.Linear(self.feature_dims, out_channels)

    def forward(self, x):#x->[2, 4, T, 19]
        gcn = self.gcn1(x)
        res = self.residual(x)
        gcn_res = gcn + res
        B, C, T, V = gcn_res.size()
        tcn = self.embed(gcn_res.permute(0, 3, 2, 1)).contiguous().view(B*V, T, -1)
        for i in range(self.num_times):
            memory = tcn
            tcn = self.tcn1[i](tcn)
            tcn += memory
        y = self.linear(tcn).contiguous().view(B, -1, T, V)
        y = self.relu(y)
        return y

class Embedding_module(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False):
        super(Embedding_module, self).__init__()
        self.out_ch = out_channels
        self.linear = nn.Linear(in_channels, out_channels, bias=bias)
    
    def forward(self, x):
        return self.linear(x) / sqrt(self.out_ch)
