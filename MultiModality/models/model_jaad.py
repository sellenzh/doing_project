import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_angles(position, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return position * angle_rates

def postional_encoding3d(position, d_model, device):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]
    return torch.tensor(pos_encoding, dtype=torch.float32).to(device)

def postional_encoding4d(position, d_model, device):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...][np.newaxis, ...]
    return torch.tensor(pos_encoding, dtype=torch.float32).to(device)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, d_input=None):
        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % num_heads == 0
        if d_input == None:
            d_input = d_model
        self.depth = d_model // num_heads

        self.q_w = nn.Linear(d_input, d_model, bias=False)
        self.k_w = nn.Linear(d_input, d_model, bias=False)
        self.v_w = nn.Linear(d_input, d_model, bias=False)

        self.dense = nn.Linear(d_model, d_model)

    def scaled_pot_product_attention(self, q, k, v):
        matmul_qk = torch.matmul(q, k.transpose(-1, -2))
        scaled_attention_logit = matmul_qk / np.sqrt(self.depth)
        return torch.matmul(nn.Softmax(dim=-1)(scaled_attention_logit), v)

    def split_heads(self, x, batch_size):
        return x.view(batch_size, -1, self.num_heads, self.depth).transpose(-1, -2)

    def forward(self, query, key, value):
        batch_size = query.shape[0]
        second_dim = query.shape[1]
        q = self.split_heads(self.q_w(query), batch_size)
        k = self.split_heads(self.k_w(key), batch_size)
        v = self.split_heads(self.v_w(value), batch_size)

        scaled_attention = self.scaled_pot_product_attention(q, k, v)
        concat_attention = scaled_attention.transpose(1, 2).contiguous().view(batch_size, second_dim, -1, self.d_model)
        return self.dense(concat_attention)


class FFN(nn.Module):
    def __init__(self, d_model, hidden_dim):
        super(FFN, self).__init__()
        self.layer1 = nn.Linear(d_model, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, d_model)
        self.relu = nn.ReLU()
    def forward(self, x):
        return self.layer2(self.relu(self.layer1(x)))

class Encoder(nn.Module):
    def __init__(self, d_model, num_heads, dff, rate=0.3, d_input=None):
        super(Encoder, self).__init__()
        self.mha = MultiHeadAttention(d_model, num_heads, d_input)
        self.ffn = FFN(d_model, dff)
        self.layernorm1 = nn.LayerNorm(normalized_shape=d_model, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(normalized_shape=d_model, eps=1e-6)
        self.dropout = nn.Dropout(rate)
    def forward(self, x, y=None):
        y = x if y is None else y
        att_output = self.dropout(self.mha(x, y, y)).squeeze()
        output1 = self.layernorm1(att_output + x)
        ffn_output = self.dropout(self.ffn(output1))
        return self.layernorm2(output1 + ffn_output)

class MLP(nn.Module):
    def __init__(self, dimensions, out_dims=None, rate=0.3):
        super(MLP, self).__init__()
        out_dims = dimensions if out_dims is None else out_dims
        self.layer1 = nn.Linear(dimensions, dimensions * 3)
        self.layer2 = nn.Linear(dimensions * 3, dimensions * 3)
        self.layer3 = nn.Linear(dimensions * 3, out_dims)
        self.ln1 = nn.LayerNorm(dimensions * 3)
        self.ln2 = nn.LayerNorm(dimensions * 3)
        self.ln3 = nn.LayerNorm(out_dims)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(rate)
    def forward(self, x):
        y = self.dropout(self.relu(self.ln1(self.layer1(x))))
        y = self.dropout(self.relu(self.ln2(self.layer2(y))))
        return self.dropout(self.relu(self.ln3(self.layer3(y))))

class Gate(nn.Module):
    def __init__(self, dims):
        super(Gate, self).__init__()
        self.fc1 = MLP(dims)
        self.fc2 = MLP(dims)
        self.sig = nn.Sigmoid()
    def forward(self, x, y):
        z1 = self.fc1(x)
        z2 = self.fc2(y)
        z = self.sig(torch.add(z1, z2))
        return torch.add(x.mul(z), y.mul(1 - z))

class Gate4(nn.Module):
    def __init__(self, dims):
        super(Gate4, self).__init__()
        self.g1 = Gate(dims)
        self.g2 = Gate(dims)
        self.g3 = Gate(dims)
        self.resize = MLP(dimensions=dims * 3, out_dims=dims)
    def forward(self, x, y1, y2, y3):
        z1 = self.g1(x, y1)
        z2 = self.g2(x, y2)
        z3 = self.g3(x, y3)
        z = torch.cat((z1, z2, z3), dim=-1)
        res = self.resize(z)
        return res

class Time_att(nn.Module):
    def __init__(self, dims, time=32):
        super(Time_att, self).__init__()
        self.linear1 = nn.Linear(dims, dims, bias=False)
        self.linear2 = nn.Linear(dims, 1, bias=False)
        self.time = nn.Linear(time, 1)
    def forward(self, x):
        y = self.linear1(x.contiguous())
        y = self.linear2(torch.tanh(y))
        beta = F.softmax(y, dim=-1)
        c = beta * x
        return self.time(c.transpose(-1, -2)).squeeze()

class VelConv(nn.Module):
    def __init__(self, dims, out_dims=None):
        super(VelConv, self).__init__()
        out_dims = dims if out_dims is None else out_dims
        self.layers = nn.Sequential(
            nn.Conv1d(in_channels=dims, out_channels=out_dims, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(out_dims), nn.ReLU())
    def forward(self, x):
        return self.layers(x)

class ImgConv(nn.Module):
    def __init__(self, dims):
        super(ImgConv, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(dims, dims, kernel_size=3, padding=1, stride=1, bias=True),
            nn.BatchNorm2d(dims), nn.ReLU()
        )
    def forward(self, img):
        return self.layers(img)

class ImgResize(nn.Module):
    def __init__(self, times, d_model):
        super(ImgResize, self).__init__()
        self.d_model = d_model
        self.times = times
        self.h, self.w = 192, 64
        self.layers = nn.Sequential(
            nn.Linear(self.w * self.h, times * 3), nn.LayerNorm(times * 3), nn.ReLU(),
            nn.Linear(times * 3, times * 3), nn.LayerNorm(times * 3), nn.ReLU(),
            nn.Linear(times * 3, times), nn.LayerNorm(times), nn.ReLU()
        )
    def forward(self, img):
        #img: b, c, h, w
        b = img.shape[0]
        return self.layers(img.view(b, -1, self.h * self.w)).transpose(-1, -2)


class DecouplingGCN(nn.Module):
    def __init__(self, d_model, num_layers, sub, groups, device):
        super(DecouplingGCN, self).__init__()
        self.num_layers = num_layers
        self.sub = sub
        self.groups = groups
        self.d_model = d_model
        
        self.para = nn.Parameter(torch.ones(3, 1, 19, 19, requires_grad=True, dtype=torch.float32).repeat(1, self.groups, 1, 1), requires_grad=True)
        nn.init.kaiming_normal_(self.para)

        self.bn = nn.BatchNorm2d(d_model * self.sub)
        self.bn1 = nn.BatchNorm2d(d_model)
        self.relu = nn.ReLU()
        self.linear_weight = nn.Parameter(torch.ones(d_model, d_model * self.sub, requires_grad=True, device=device), requires_grad=True)
        nn.init.kaiming_normal_(self.linear_weight)
        self.linear_bias = nn.Parameter(torch.ones(1, d_model * self.sub, 1, 1, requires_grad=True, device=device), requires_grad=True)
        nn.init.kaiming_normal_(self.linear_bias)
    def L2_norm(self, A):
        A_norm = torch.norm(A, 2, dim=-1, keepdim=True) + 1e-4
        return A / A_norm
    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        learn_A = self.para.repeat(1, self.d_model // self.groups, 1, 1)
        norm_learn_A = torch.cat([self.L2_norm(learn_A[0:1, ...]),
                                self.L2_norm(learn_A[1:2, ...]),
                                self.L2_norm(learn_A[2:3, ...])], 0)
        y = torch.einsum('nctw,cd->ndtw', (x, self.linear_weight)).contiguous()
        y = self.bn(y + self.linear_bias)
        n, kc, t, v = y.size()
        y = y.view(n, self.sub, kc // self.sub, t, v)
        y = torch.einsum('nkctv,kcvw->nctw', (y, norm_learn_A))
        return self.relu(self.bn1(y + x)).permute(0, 2, 3, 1)

class KpsResize(nn.Module):
    def __init__(self):
        super(KpsResize, self).__init__()
        self.layers = MLP(dimensions=19, out_dims=1)
    def forward(self, kps):
        #b, c, t, v -> b, t, c
        return self.layers(kps).squeeze()

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.sigma1 = nn.Parameter(torch.ones(1, 1, requires_grad=True).to(device), requires_grad=True)
        nn.init.kaiming_normal_(self.sigma1)
        self.sigma2 = nn.Parameter(torch.ones(1, 1, requires_grad=True).to(device), requires_grad=True)
        nn.init.kaiming_normal_(self.sigma2)

        self.d_model = args.d_model
        self.num_layers = args.num_layers
        self.bbox = args.bbox_input
        self.vel = args.vel_input
        self.kps = args.kps_input
        self.num_heads = args.num_heads
        self.dff = args.dff
        self.posi_en_dim = args.encoding_dims
        self.gcn = args.gcn
        self.sub = args.sub
        self.groups = args.groups
        self.times_num = args.times_num

        self.bbox_posi_enc = postional_encoding3d(self.times_num, self.d_model, device)
        self.bbox_embedding = nn.Linear(self.bbox, self.d_model)
        self.bbox_self_layers = nn.ModuleList()
        self.bbox_vel_layers = nn.ModuleList()
        self.bbox_img_layers = nn.ModuleList()
        self.bbox_kps_layers = nn.ModuleList()
        self.gates = nn.ModuleList()
        #self.gate = nn.ModuleList()

        self.vel_posi_enc = postional_encoding3d(self.times_num, self.d_model, device)
        self.vel_embedding = nn.Linear(self.vel, self.d_model)
        self.vel_layers = nn.ModuleList()

        self.kps_embedding = nn.Linear(self.kps, self.d_model)
        self.kps_posi_enc = postional_encoding4d(self.times_num, self.d_model, device)
        self.spatial = nn.ModuleList()
        self.time = nn.ModuleList()
        self.kps_resize = nn.ModuleList()

        self.img_embedding = nn.Conv2d(4, self.d_model, kernel_size=3, padding=1)
        self.img_conv = nn.ModuleList()
        self.img_resize = nn.ModuleList()

        for _ in range(self.num_layers):
            self.bbox_self_layers.append(Encoder(self.d_model, self.num_heads, self.dff))

            self.bbox_vel_layers.append(Encoder(self.d_model, self.num_heads, self.dff))
            self.bbox_img_layers.append(Encoder(self.d_model, self.num_heads, self.dff))
            self.bbox_kps_layers.append(Encoder(self.d_model, self.num_heads, self.dff))

            self.gates.append(Gate4(dims=self.d_model))
            #self.gate.append(Gate(dims=self.d_model))

            self.vel_layers.append(VelConv(self.d_model))

            if self.gcn:
                self.spatial.append(DecouplingGCN(self.d_model, self.num_layers, self.sub, self.groups, device))
            else:
                self.spatial.append(Encoder(self.d_model, self.num_heads, self.dff))
            self.time.append(Encoder(self.d_model, self.num_heads, self.dff))

            self.kps_resize.append(KpsResize())

            self.img_conv.append(ImgConv(self.d_model))
            self.img_resize.append(ImgResize(self.times_num, self.d_model))

        self.time_att = Time_att(self.d_model)
        self.cross = MLP(self.d_model, 4)

        self.cp2f = MLP(4, self.d_model)

        self.pool1d = nn.AdaptiveAvgPool1d(1)
        self.pool2d = nn.AdaptiveAvgPool2d(1)
        self.gate_last = Gate4(dims=self.d_model)
        self.linear = nn.Linear(self.d_model, 4)
        self.relu = nn.ReLU()
        self.last = nn.Linear(4, 1)
        self.sig = nn.Sigmoid()

    def forward(self, kps, img, bbox, vel):
        '''
        :kps        :[b, 4, 32, 19]
        :img        :[b, 4, 192, 64]
        :bbox       :[b, 4, 32]
        :vel        :[b, 2, 32]
        '''
        seq_len = bbox.shape[-1]

        bbox = self.bbox_embedding(bbox.transpose(-1, -2))#[b, 32, 256]
        bbox += self.bbox_posi_enc[:, :seq_len, :]

        vel = self.vel_embedding(vel.transpose(-1, -2)) #[b, 32, 256]
        vel += self.vel_posi_enc[:, :seq_len, :]

        kps = self.kps_embedding(kps.transpose(-1, 1))#[b, 256, 32, 19]
        kps += self.kps_posi_enc[:, :, :seq_len, :]
        kps = kps.transpose(-1, 1)

        img = self.img_embedding(img)#[b, 256, 192, 64]

        for i in range(self.num_layers):
            bbox = self.bbox_self_layers[i](bbox)
            vel = self.vel_layers[i](vel.transpose(-1, -2)).transpose(-1, -2)#b, 32, 256
            bbox_vel = self.bbox_vel_layers[i](bbox, vel)
            img = self.img_conv[i](img)#b, 256, 192, 64
            bbox_img = self.bbox_img_layers[i](bbox, self.img_resize[i](img))
            kps = self.time[i](self.spatial[i](kps.permute(0, 2, 3, 1)).permute(0, 2, 1, 3)).permute(0, 3, 2, 1)
            bbox_kps = self.bbox_kps_layers[i](bbox, self.kps_resize[i](kps).transpose(-1, -2))
            bbox = self.gates[i](bbox, bbox_img, bbox_vel, bbox_kps)
        
        pred_point = self.cross(self.time_att(bbox))

        feature = self.cp2f(pred_point)#b,256
        vel_pool = self.pool1d(vel.transpose(-1, -2)).squeeze()# b, 32, 256
        img_pool = self.pool2d(img).squeeze() #b, 256, 192, 64
        kps_pool = self.pool2d(kps).squeeze() #b, 256, 32, 19
        #y = torch.cat((feature, vel_pool, img_pool, kps_pool), dim=-1)
        y = self.gate_last(feature, vel_pool, img_pool, kps_pool)
        y = self.relu(self.linear(y))
        return self.sig(self.last(y)), pred_point, self.sigma1, self.sigma2

'''import argparse
parser = argparse.ArgumentParser('pedestrian model')
parser.add_argument('--logdir', type=str, default='./log/JAAD', help='save path')
parser.add_argument('--device', type=str, default='cpu', help='choose device.')
parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train.')
parser.add_argument('--lr', type=int, default=0.001, help='learning rate to train.')
parser.add_argument('--data_path', type=str, default='./data/JAAD', help='data path')
parser.add_argument('--num_workers', type=int, default=0, help='Number of workers for the dataloader.')
parser.add_argument('--forcast', type=bool, default=False, help='Use the human pose forcasting data.')
parser.add_argument('--jaad_path', type=str, default='./JAAD')
parser.add_argument('--balance', type=bool, default=True, help='Balnce or not the data set')
parser.add_argument('--bh', type=str, default='all', help='all or bh, if use all samples or only samples with behaevior labers')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--batch_size', type=int, default=64, help='size of batch.')'''
'''parser.add_argument('--d_model', type=int, default=128, help='the dimension after embedding.')
parser.add_argument('--num_layers', type=int, default=4, help='the number of layers.')
parser.add_argument('--dff', type=int, default=256, help='the number of the units.')
parser.add_argument('--num_heads', type=int, default=8, help='number of the heads of the multi-head model.')
parser.add_argument('--encoding_dims', type=int, default=32, help='dimension of the time.')
parser.add_argument('--bbox_input', type=int, default=4, help='dimension of bbox.')
parser.add_argument('--vel_input', type=int, default=2, help='dimension of velocity.')
parser.add_argument('--kps_input', type=int, default=4, help='the dimension of keypoints.')
parser.add_argument('--gcn', type=bool, default=True, help='Use Graph conv networks else use spatial attention.')
parser.add_argument('--times_num', type=int, default=32, help='the number of time dimension.')
parser.add_argument('--sub', type=int, default=3, help='the number of sub matrix')
parser.add_argument('--groups', type=int, default=8, help='the groups of the learn matrix.')
args = parser.parse_args()

model = Model(args)
kps = torch.randn(size=(16, 4, 32, 19))
img = torch.randn(size=(16, 4, 192, 64))
bbox = torch.randn(size=(16, 4, 32))
vel = torch.randn(size=(16, 2, 32))
y, pred_point, sigm1, sigm2 = model(kps, img, bbox, vel)
print('end!')'''
