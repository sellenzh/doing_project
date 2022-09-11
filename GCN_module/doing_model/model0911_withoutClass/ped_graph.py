from torch import nn
import numpy as np
import math
import torch
from entmax import entmax15
import torch.nn.functional as F

def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)

def conv_init(conv):
    if conv.weight is not None:
        nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)

class PedModel(nn.Module):
    def __init__(self, n_clss=1):
        super(PedModel, self).__init__()
        self.ch_img, self.ch_vel, self.ch1, self.ch2 = 4, 2, 32, 64
        self.n_clss = n_clss
        self.nodes = 19
        self.heads = 8
        self.rate = 0.3
        #cross attention args
        self.heads = 8
        self.layers_num = 5
        self.dropout = 0.3

        self.data_bn = nn.BatchNorm1d(self.ch_img * self.nodes)
        bn_init(self.data_bn, 1)
        self.drop = nn.Dropout(0.25)
        A = np.stack([np.eye(self.nodes)] * 3, axis=0)

        self.img1 = nn.Sequential(
            nn.Conv2d(self.ch_img, self.ch1, kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.ch1), nn.SiLU())

        self.vel1 = nn.Sequential(
            nn.Conv1d(self.ch_vel, self.ch1, 3, bias=False), nn.BatchNorm1d(self.ch1), nn.SiLU())
        # ----------------------------------------------------------------------------------------------------

        self.img2 = nn.Sequential(
            nn.Conv2d(self.ch1, self.ch1, kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.ch1), nn.SiLU())

        self.vel2 = nn.Sequential(
            nn.Conv1d(self.ch1, self.ch1, 3, bias=False),
            nn.BatchNorm1d(self.ch1), nn.SiLU())
        # ----------------------------------------------------------------------------------------------------

        self.img3 = nn.Sequential(
            nn.Conv2d(self.ch1, self.ch2, kernel_size=2, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.ch2), nn.SiLU())

        self.vel3 = nn.Sequential(
            nn.Conv1d(self.ch1, self.ch2, kernel_size=2, bias=False),
            nn.BatchNorm1d(self.ch2), nn.SiLU())

        self.layer1 = GCN_TAT_unit(self.ch_img, self.ch1, A, adaptive=False)
        self.layer2 = GCN_TAT_unit(self.ch1, self.ch2, A)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.att = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.ch2, self.ch2, bias=False),
            nn.BatchNorm1d(self.ch2),
            nn.Sigmoid()
        )
        self.linear = nn.Linear(self.ch2, self.n_clss)
        nn.init.normal_(self.linear.weight, 0, math.sqrt(2. / self.n_clss))

        self.pool_sig_2d = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Sigmoid()
        )

        self.pool_sig_1d = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Sigmoid()
        )
        #---------------------------------------------------------------------------------
        self.attention1 = nn.ModuleList()
        for _ in range(self.layers_num):
            self.attention1.append(
                Encoder(inputs=self.ch1, heads=self.heads, hidden=self.ch1 * 3, a_dropout=self.dropout,
                        f_dropout=self.dropout))
        self.attention2 = nn.ModuleList()
        for _ in range(self.layers_num):
            self.attention2.append(
                Encoder(inputs=self.ch2, heads=self.heads, hidden=self.ch2 * 3, a_dropout=self.dropout,
                        f_dropout=self.dropout))
        # ---------------------------------------------------------------------------------
        self.vel_nodes = nn.Sequential(
            nn.Linear(1, self.nodes * 3, bias=False),
            # nn.LayerNorm(self.nodes * 3), nn.SiLU(),
            nn.BatchNorm2d(self.ch1), nn.ReLU(),
            nn.Linear(self.nodes * 3, self.nodes * 3, bias=False),
            # nn.LayerNorm(self.nodes * 3), nn.SiLU(),
            nn.BatchNorm2d(self.ch1), nn.ReLU(),
            nn.Linear(self.nodes * 3, self.nodes, bias=False),
            # nn.LayerNorm(self.nodes), nn.SiLU(),
            nn.BatchNorm2d(self.ch1), nn.ReLU()
        )
        self.linear1 = nn.Sequential(
            nn.Conv1d(2, self.ch1 * 2, kernel_size=1, bias=False),
            nn.BatchNorm1d(self.ch1 * 2), nn.ReLU(),
            nn.Conv1d(self.ch1 * 2, self.ch1 * 2, kernel_size=1, bias=False),
            nn.BatchNorm1d(self.ch1 * 2), nn.ReLU(),
            nn.Conv1d(self.ch1 * 2, self.ch1, kernel_size=1, bias=False),
            nn.BatchNorm1d(self.ch1), nn.ReLU()
        )
        self.linear2 = nn.Sequential(
            nn.Conv2d(self.ch1, self.ch2 * 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.ch2 * 2), nn.ReLU(),
            nn.Conv2d(self.ch2 * 2, self.ch2 * 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.ch2 * 2), nn.ReLU(),
            nn.Conv2d(self.ch2 * 2, self.ch2, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.ch2), nn.ReLU()
        )
        # ---------------------------------------------------------------------------------
        self.fc_layers1 = nn.Sequential(
            nn.Linear(self.ch1, self.ch1 * 3),
            nn.LayerNorm(self.ch1 * 3), nn.ReLU(),
            nn.Linear(self.ch1 * 3, self.ch1 * 3),
            nn.LayerNorm(self.ch1 * 3), nn.ReLU(),
            nn.Linear(self.ch1 * 3, self.ch1),
            nn.LayerNorm(self.ch1), nn.ReLU()
        )  # way2
        self.fc_layers2 = nn.Sequential(
            nn.Linear(self.ch1, self.ch1 * 3),
            nn.LayerNorm(self.ch1 * 3), nn.ReLU(),
            nn.Linear(self.ch1 * 3, self.ch1 * 3),
            nn.LayerNorm(self.ch1 * 3), nn.ReLU(),
            nn.Linear(self.ch1 * 3, self.ch1),
            nn.LayerNorm(self.ch1), nn.ReLU()
        )  # way2
        self.fc_layers3 = nn.Sequential(
            nn.Linear(self.ch2, self.ch2 * 3),
            nn.LayerNorm(self.ch2 * 3), nn.ReLU(),
            nn.Linear(self.ch2 * 3, self.ch2 * 3),
            nn.LayerNorm(self.ch2 * 3), nn.ReLU(),
            nn.Linear(self.ch2 * 3, self.ch2),
            nn.LayerNorm(self.ch2), nn.ReLU()
        )  # way2
        self.fc_layers4 = nn.Sequential(
            nn.Linear(self.ch2, self.ch2 * 3),
            nn.LayerNorm(self.ch2 * 3), nn.ReLU(),
            nn.Linear(self.ch2 * 3, self.ch2 * 3),
            nn.LayerNorm(self.ch2 * 3), nn.ReLU(),
            nn.Linear(self.ch2 * 3, self.ch2),
            nn.LayerNorm(self.ch2), nn.ReLU()
        )  # way2
        self.fc_bn1 = nn.BatchNorm2d(self.ch1)
        self.fc_bn2 = nn.BatchNorm2d(self.ch2)
        self.fc_act = nn.ReLU()
        self.fc_dropout = nn.Identity()
        #


        # ---------------------------------------------------------------------------------

    def cross_att1(self, pose, info):
        N, C, T, V = pose.size()
        pose = pose.permute(0, 2, 3, 1).contiguous().view(N * T, V, -1)
        info = info.permute(0, 2, 3, 1).contiguous().view(N * T, V, -1)

        for i in range(self.layers_num):
            pose_att = self.attention1[i](pose, info)
            pose_att += pose
        return pose_att.view(N, T, V, -1).permute(0, 3, 1, 2).contiguous()

    def cross_att2(self, pose, info):
        N, C, T, V = pose.size()
        pose = pose.permute(0, 2, 3, 1).contiguous().view(N * T, V, -1)
        info = info.permute(0, 2, 3, 1).contiguous().view(N * T, V, -1)

        for i in range(self.layers_num):
            pose_att = self.attention2[i](pose, info)
            pose_att += pose
        return pose_att.view(N, T, V, -1).permute(0, 3, 1, 2).contiguous()

    def FC1(self, x):
        x = self.fc_layers1(x.permute(0, 2, 3, 1))
        y = self.fc_act(self.fc_bn1(x.permute(0, 3, 1, 2)))
        y = self.fc_dropout(y)
        return y
    def FC2(self, x):
        x = self.fc_layers2(x.permute(0, 2, 3, 1))
        y = self.fc_act(self.fc_bn1(x.permute(0, 3, 1, 2)))
        y = self.fc_dropout(y)
        return y
    def FC3(self, x):
        x = self.fc_layers3(x.permute(0, 2, 3, 1))
        y = self.fc_act(self.fc_bn2(x.permute(0, 3, 1, 2)))
        y = self.fc_dropout(y)
        return y
    def FC4(self, x):
        x = self.fc_layers4(x.permute(0, 2, 3, 1))
        y = self.fc_act(self.fc_bn2(x.permute(0, 3, 1, 2)))
        y = self.fc_dropout(y)
        return y

    def gated1(self, x, y):
        z1 = self.FC1(x)
        z2 = self.FC2(y)
        z = torch.sigmoid(torch.add(z1, z2))
        #z = 0
        res = torch.add(torch.mul(x, 1 - z), torch.mul(y, z))
        return res

    def gated2(self, x, y):
        z1 = self.FC3(x)
        z2 = self.FC4(y)
        z = torch.sigmoid(torch.add(z1, z2))
        #z = 0
        res = torch.add(torch.mul(x, 1 - z), torch.mul(y, z))
        return res

    def expand_vel(self, vel):
        ve = self.linear1(vel).unsqueeze(-1)
        ve1 = self.vel_nodes(ve)
        ve2 = self.linear2(ve1)
        return ve1, ve2


    def forward(self, pose, frame, vel):
        N, C, T, V = pose.shape
        kp = pose.permute(0, 1, 3, 2).contiguous().view(N, C * V, T)
        kp = self.data_bn(kp)
        kp = kp.view(N, C, V, T).permute(0, 1, 3, 2).contiguous()

        f1 = self.img1(frame)  # [2, 32, 190, 62]
        v1 = self.vel1(vel)  # [2, 32, T-2]

        pose = self.layer1(kp)
        f1 = self.img2(f1)
        v1 = self.vel2(v1)
        pose = pose.mul(self.pool_sig_1d(v1).unsqueeze(-1))

        ve1, ve2 = self.expand_vel(vel)
        pose_att1 = self.cross_att1(pose, ve1[:, :, -T:, :])
        pose = self.gated1(pose, pose_att1)

        pose = self.layer2(pose)
        f1 = self.img3(f1)
        pose = pose.mul(self.pool_sig_2d(f1))
        v1 = self.vel3(v1)
        pose = pose.mul(self.pool_sig_1d(v1).unsqueeze(-1))

        pose_att2 = self.cross_att2(pose, ve2[:, :, -T:, :])
        pose = self.gated2(pose, pose_att2)

        pose = self.gap(pose).squeeze(-1)
        pose = pose.squeeze(-1)
        y = self.att(pose).mul(pose) + pose
        y = self.drop(y)
        y = self.linear(y)

        return y


class ConvResidual(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1):
        super(ConvResidual, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        y = self.conv(x)
        return self.bn(y)


class DecouplingGcn(nn.Module):
    def __init__(self, in_channels, out_channels, A, adaptive) -> None:
        super(DecouplingGcn, self).__init__()
        self.in_ch = in_channels
        self.out_ch = out_channels
        self.subnet = A.shape[0]
        self.groups = 8
        self.adaptive = adaptive

        self.linear = nn.Linear(self.in_ch, self.out_ch)
        self.DecoupleA = nn.Parameter(torch.tensor(np.reshape(A.astype(np.float32), [
            3, 1, 19, 19]), dtype=torch.float32, requires_grad=True).repeat(1, self.groups, 1, 1), requires_grad=True)

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
            in_channels, out_channels * self.subnet, requires_grad=True), requires_grad=True)#, device='cuda'
        nn.init.normal_(self.Linear_weight, 0, math.sqrt(
            0.5 / (out_channels * self.subnet)))

        self.Linear_bias = nn.Parameter(torch.zeros(
            1, out_channels * self.subnet, 1, 1, requires_grad=True), requires_grad=True)#, device='cuda'
        nn.init.constant_(self.Linear_bias, 1e-6)

    def L2_norm(self, A):
        A_norm = torch.norm(A, 2, dim=1, keepdim=True) + 1e-4  # N,1,V
        A = A / A_norm
        return A

    def forward(self, x):
        learn_A = self.DecoupleA.repeat(1, self.out_ch // self.groups, 1, 1)  # learn_A -> [3, 32, 19, 19]
        norm_learn_A = torch.cat([self.L2_norm(learn_A[0:1, ...]),
                                  self.L2_norm(learn_A[1:2, ...]),
                                  self.L2_norm(learn_A[2:3, ...])],
                                 0)
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


class GCN_TAT_unit(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True, adaptive=True):
        super(GCN_TAT_unit, self).__init__()
        self.feature_dims = out_channels * 3
        self.num_heads = 8
        self.hidden_dims = out_channels * 6
        self.attention_dropout = 0.3
        self.tat_times = 5

        self.res = residual
        self.gcn = DecouplingGcn(in_channels, out_channels, A, adaptive)

        self.embed = Embedding_module(out_channels, self.feature_dims)
        self.tat = nn.ModuleList(
            Encoder(self.feature_dims, self.num_heads, self.hidden_dims, self.attention_dropout) for _ in
            range(self.tat_times))
        self.relu = nn.ReLU(inplace=True)
        if not residual:
            self.residual = lambda x: 0
        elif in_channels == out_channels and stride == 1:
            self.residual = lambda x: x
        else:
            #self.residual = lambda x: x
            self.residual = ConvResidual(in_channels, out_channels, kernel_size=1, stride=stride)

        self.linear = nn.Linear(self.feature_dims, out_channels)

    def forward(self, x):  # x->[2, 4, T, 19]
        gcn = self.gcn(x)
        res = self.residual(x)
        x = gcn + res
        B, C, T, V = x.size()
        tcn = self.embed(x.permute(0, 3, 2, 1)).contiguous().view(B * V, T, -1)
        for i in range(self.tat_times):
            memory = tcn
            tcn = self.tat[i](tcn)
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
        return self.linear(x) / math.sqrt(self.out_ch)

class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout=None):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(p=dropout) if dropout is not None else nn.Identity()

    def forward(self, q, k, v):
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))

        attn = entmax15(scores, dim=-1)
        y = torch.matmul(attn, v)
        return self.dropout(y)


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, inputs, a_dropout=None, f_dropout=None):
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

    def forward(self, q, k, v):
        bs = q.size(0)
        q = self.linear_q(q).view(bs, -1, self.heads, self.hidden).transpose(1, 2)
        k = self.linear_k(k).view(bs, -1, self.heads, self.hidden).transpose(1, 2)
        v = self.linear_v(v).view(bs, -1, self.heads, self.hidden).transpose(1, 2)

        out = self.attention(q, k, v).transpose(1, 2).contiguous()
        out = out.view(bs, -1, self.inputs)
        return self.dropout(self.output(out))


class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x * (torch.tanh(F.softplus(x)))
        return x


class FeedForwardNet(nn.Module):
    def __init__(self, inputs, hidden, dropout):
        super(FeedForwardNet, self).__init__()
        self.upscale = nn.Linear(inputs, hidden)
        self.activation = Mish()
        self.downscale = nn.Linear(hidden, inputs)
        self.dropout = nn.Dropout(p=dropout) if dropout is not None else nn.Identity()

    def forward(self, x):
        x = self.upscale(x)
        x = self.activation(x)
        x = self.downscale(x)
        return self.dropout(x)


class EncoderLayer(nn.Module):
    def __init__(self, inputs, heads, hidden, a_dropout=None, f_dropout=None):
        super(EncoderLayer, self).__init__()

        self.attention = MultiHeadAttention(heads, inputs, a_dropout=a_dropout, f_dropout=f_dropout)
        self.attention_norm = nn.LayerNorm(inputs)
        self.feedforward = FeedForwardNet(inputs, hidden, dropout=f_dropout)
        self.feedforward_norm = nn.LayerNorm(inputs)

    def forward(self, x, z=None):
        z = x if z is None else z
        y = self.attention(x, z, z)
        y = self.attention_norm(y)
        x = x + y
        y = self.feedforward(x)
        y = self.feedforward_norm(y)
        x = x + y
        return x


class Encoder(nn.Module):
    def __init__(self, inputs, heads, hidden, a_dropout=None, f_dropout=None):
        super(Encoder, self).__init__()

        self.norm = nn.LayerNorm(inputs)
        self.layers = EncoderLayer(inputs, heads, hidden, a_dropout=a_dropout, f_dropout=f_dropout)

    def forward(self, x, y=None):
        x = self.layers(x, y)
        return self.norm(x)


'''model = PedModel(n_clss=3)
img = torch.randn(size=(16, 4, 192, 64))
vel = torch.randn(size=(16, 2, 62))
pose = torch.randn(size=(16, 4, 62, 19))
y = model(pose, img, vel)
print(y.size())'''
