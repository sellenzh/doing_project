import torch
import numpy as np
from torch import nn
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"

def get_angles(position, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return position * angle_rates

def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]
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

    def scaled_pot_product_attention(self, q, k, v, mask=None):
        matmul_qk = torch.matmul(q, k.transpose(-1, -2))
        scaled_attention_logit = matmul_qk / np.sqrt(self.depth)
        if mask is not None:
            scaled_attention_logit += (mask * -1e9)
        return torch.matmul(nn.Softmax(dim=-1)(scaled_attention_logit), v)

    def split_heads(self, x, batch_size):
        return x.view(batch_size, -1, self.num_heads, self.depth).transpose(1, 2)

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        q = self.split_heads(self.q_w(query), batch_size)
        k = self.split_heads(self.k_w(key), batch_size)
        v = self.split_heads(self.v_w(value), batch_size)

        scaled_attention = self.scaled_pot_product_attention(q, k, v, mask)
        concat_attention = scaled_attention.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.dense(concat_attention)


class FFN(nn.Module):
    def __init__(self, d_model, hidden_dim):
        super(FFN, self).__init__()
        self.layer1 = nn.Linear(d_model, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, d_model)
        self.activation = nn.ReLU()

    def forward(self, x):
        y = self.layer1(x)
        return self.activation(self.layer2(y))


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dff, rate, d_input=None):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads, d_input)
        self.ffn = FFN(d_model, dff)

        self.layer1norm = nn.LayerNorm(normalized_shape=d_model, eps=1e-6)
        self.layer2norm = nn.LayerNorm(normalized_shape=d_model, eps=1e-6)

        self.dropout1 = nn.Dropout(rate)
        self.dropout2 = nn.Dropout(rate)

    def forward(self, x, y=None, mask=None):
        y = x if y is not None else y
        att_output = self.dropout1(self.mha(x, y, y, mask))
        output1 = self.layer1norm(y + att_output)

        ffn_output = self.dropout2(self.ffn(output1))
        return self.layer2norm(output1 + ffn_output)


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


class MLP(nn.Module):
    def __init__(self, dimensions, dropout_rate=0.3):
        super(MLP, self).__init__()

        self.layer1 = nn.Linear(dimensions, dimensions * 3)
        self.layer2 = nn.Linear(dimensions * 3, dimensions * 3)
        self.layer3 = nn.Linear(dimensions * 3, dimensions)
        self.ln1 = nn.LayerNorm(dimensions * 3)
        self.ln2 = nn.LayerNorm(dimensions * 3)
        self.ln3 = nn.LayerNorm(dimensions)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        y = self.dropout(self.act(self.ln1(self.layer1(x))))
        y = self.dropout(self.act(self.ln2(self.layer2(y))))
        return self.dropout(self.act(self.ln3(self.layer3(y))))


class Time_att(nn.Module):
    def __init__(self, dims, time=16):
        super(Time_att, self).__init__()
        self.linear1 = MLP(dims)
        self.linear2 = nn.Linear(dims, 1, bias=False)
        self.time = nn.Linear(time, 1)

    def forward(self, x):
        x = x.contiguous()
        y = self.linear2(torch.tanh(self.linear1(x)))

        beta = F.softmax(y, dim=1)
        c = beta * x
        c = self.time(c.transpose(-1, -2)).squeeze()
        return c


class ConvLayers(nn.Module):
    def __init__(self, dims, out_dims=None):
        super(ConvLayers, self).__init__()

        out_dims = dims if out_dims is None else out_dims
        self.layers = nn.Sequential(
            nn.Conv1d(in_channels=dims, out_channels=out_dims, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(out_dims), nn.ReLU(),
            nn.Conv1d(in_channels=out_dims, out_channels=out_dims, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(out_dims), nn.ReLU(),
            nn.Conv1d(in_channels=out_dims, out_channels=out_dims, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(out_dims), nn.ReLU()
        )

    def forward(self, x):
        return self.layers(x)


class Model(nn.Module):
    def __init__(self, num_layers, d_model, bbox_input, speed_input, num_heads, dff, pe_input, pe_target, rate=0.1):
        super(Model, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.pe_target = pe_target
        self.pe_input = pe_input
        self.bbox_input = bbox_input
        self.speed_input = speed_input

        self.embedding_bbox = nn.Linear(bbox_input, d_model)
        self.embedding_vel = nn.Linear(speed_input, d_model)
        self.embedding_endpoint = nn.Linear(bbox_input, d_model)

        self.enc_layers = nn.ModuleList()
        self.fusion_weight = nn.ModuleList()
        self.cross = nn.ModuleList()
        self.conv = nn.ModuleList()
        self.gate = nn.ModuleList()

        for _ in range(self.num_layers):
            self.enc_layers.append(EncoderLayer(d_model, num_heads, dff, rate))
            self.cross.append(EncoderLayer(d_model, num_heads, dff, rate))
            self.conv.append(ConvLayers(d_model))
            self.gate.append(Gate(d_model))

        self.pos_encoding = positional_encoding(pe_input, self.d_model)
        self.pos_encoding_vel = positional_encoding(pe_input, self.d_model)
        self.pos_encoding_traj = positional_encoding(pe_target - 1, self.d_model)
        self.dropout = nn.Dropout(rate)

        self.traj_weight1 = nn.ParameterList()
        self.traj_weight2 = nn.ParameterList()
        self.traj_bias = nn.ParameterList()
        for _ in range(self.num_layers):
            #bdc,duc->buc
            self.traj_weight1.append(nn.Parameter(torch.zeros(
                pe_input, 3, d_model, requires_grad=True, device=device), requires_grad=True))
            #buc,utc->btc
            self.traj_weight2.append(nn.Parameter(torch.zeros(3, 1, d_model, requires_grad=True, device=device), requires_grad=True))
            #buc
            self.traj_bias.append(nn.Parameter(torch.zeros(
                1, 3, 1, requires_grad=True, device=device), requires_grad=True))

        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(1)

        self.resize = nn.Linear(d_model * 2, d_model)
        self.att_t = Time_att(d_model)
        self.traj_resize = ConvLayers(dims=pe_target -1 , out_dims=1)
        self.endp_fus = Gate(dims=d_model)

        self.final_layer = nn.Linear(d_model, bbox_input)
        self.linear = nn.Linear(d_model, 4)
        self.dense = nn.Linear(4, 1)
        self.activation = nn.Sigmoid()

        self.sigma_cls = nn.Parameter(torch.zeros(1, 1, requires_grad=True, device=device), requires_grad=True)
        nn.init.kaiming_normal_(self.sigma_cls, mode='fan_out')
        self.sigma_reg = nn.Parameter(torch.zeros(1, 1, requires_grad=True, device=device), requires_grad=True)
        nn.init.kaiming_normal_(self.sigma_reg, mode='fan_out')

    def forward(self, data, inp_dec, combined_mask=None, dec_padding_mask=None, enc_padding_mask=None):
        x = data[:, :, :self.bbox_input]  # [64, 16, 4]
        vel = data[:, :, -self.speed_input:]  # [64, 16, 2]
        #inp_dec = inp_dec[:, -self.pe_target:, :]  # [64, 59, 4]

        seq_len = x.shape[-2]
        #traj_len = inp_dec.shape[-2]
        x = self.embedding_bbox(x)  # [64, 16, 256]
        vel = self.embedding_vel(vel)  # [64, 16, 256]
        inp_dec = self.embedding_endpoint(inp_dec)  # [64, 59, 4 -> 256]

        x += self.pos_encoding[:, :seq_len, :]
        vel += self.pos_encoding_vel[:, :seq_len, :]
        inp_dec += self.pos_encoding_traj[:, :self.pe_target - 1, :]
        traj = self.dropout(self.traj_resize(inp_dec))#[b, t, c]=>[b,1,c]

        for i in range(self.num_layers):
            sa_x = self.enc_layers[i](x, x, enc_padding_mask)
            vel = self.conv[i](vel.transpose(-1, -2)).transpose(-1, -2)
            bbox_cross = self.cross[i](x, vel)
            x = self.gate[i](sa_x, bbox_cross)

            traj1 = torch.einsum('bdc,duc->buc', (x, self.traj_weight1[i])).contiguous() + self.traj_bias[i]
            traj += torch.einsum('buc,utc->btc', (traj1, self.traj_weight2[i])).contiguous()
            # x:[64, 16, 256], traj_w: [16, 3], tarj_b: [1, 3, 1], traj: [64, 1, 256]
            traj = self.bn(self.relu(traj))

        y = self.resize(torch.cat((x, vel), dim=-1))

        y = self.att_t(y)
        y = self.endp_fus(y, traj[:, -1, :])
        y = self.relu(self.linear(y))
        y_traj = self.final_layer(traj)

        return y_traj, self.activation(self.dense(y)), self.sigma_cls, self.sigma_reg
