import torch
import numpy as np
from torch import nn
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"

#Positional encoding

def get_angles(position, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return position * angle_rates

def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)
    #apply sin to even indices in the array ; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    #apply cos to odd indices in the array : 2i + 1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]
    return torch.tensor(pos_encoding, dtype=torch.float32).to(device)

def create_look_ahead_mask(size):
    mask = torch.triu(torch.ones((size, size), dtype=torch.float32), diagonal=1)
    return mask


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, d_input=None):
        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        if d_input is None:
            d_input = d_model

        self.depth = d_model // self.num_heads
        self.q_w = nn.Linear(d_input, d_model, bias=False)
        self.k_w = nn.Linear(d_input, d_model, bias=False)
        self.v_w = nn.Linear(d_input, d_model, bias=False)

        self.dense = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, q, k, v):
        matmul_qk = torch.matmul(q, k.transpose(-1, -2))
        scaled_attention_logit = matmul_qk / np.sqrt(self.depth)
        attention_weights = nn.Softmax(dim=-1)(scaled_attention_logit)
        return torch.matmul(attention_weights, v)

    def split_heads(self, x, batch_size):
        return x.view(batch_size, -1, self.num_heads, self.depth).transpose(1, 2)

    def forward(self, q, k, v):
        batch_size = q.shape[0]

        q = self.split_heads(self.q_w(q), batch_size)  # (batch_size, num_heads, seq_len_q, depth) --- F
        k = self.split_heads(self.k_w(k), batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(self.v_w(v), batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        scaled_attention = self.scaled_dot_product_attention(q, k, v)
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
        y = self.activation(self.layer2(y))
        return y


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dff, rate, d_input=None):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads, d_input)
        self.ffn = FFN(d_model, dff)

        self.layer1norm = nn.LayerNorm(normalized_shape=d_model, eps=1e-6)
        self.layer2norm = nn.LayerNorm(normalized_shape=d_model, eps=1e-6)

        self.dropout1 = nn.Dropout(rate)
        self.dropout2 = nn.Dropout(rate)

    def forward(self, x, y=None):
        y = x if y is None else y
        att_output = self.mha(x, y, y)
        att_output = self.dropout1(att_output)
        out1 = self.layer1norm(y + att_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        out = self.layer2norm(out1 + ffn_output)
        return out


class Encoder(nn.Module):
    def __init__(self, num_layers, d_model, bbox_input, speed_input, num_heads, dff, maximum_position_encoding, rate=0.1):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        self.embedding_bbox = nn.Linear(bbox_input, d_model)
        self.embedding_vel = nn.Linear(speed_input, d_model)


        self.enc_layers = nn.ModuleList()
        self.vel_layers = nn.ModuleList()
        self.cross_bbox2vel = nn.ModuleList()
        self.cross_vel2bbox = nn.ModuleList()
        for _ in range(self.num_layers):
            self.enc_layers.append(EncoderLayer(d_model, num_heads, dff, rate))
            self.vel_layers.append(EncoderLayer(d_model, num_heads, dff, rate))
            self.cross_bbox2vel.append(EncoderLayer(d_model, num_heads, dff, rate))
            self.cross_vel2bbox.append(EncoderLayer(d_model, num_heads, dff, rate))

        self.pos_encoding = positional_encoding(maximum_position_encoding, self.d_model)
        self.pos_encoding_vel = positional_encoding(maximum_position_encoding, self.d_model)


    def forward(self, x, vel):
        seq_len = x.shape[-2]
        x = self.embedding_bbox(x)
        vel = self.embedding_vel(vel)

        x += self.pos_encoding[:, :seq_len, :]
        vel += self.pos_encoding_vel[:, :seq_len, :]

        for i in range(self.num_layers):
            x = self.enc_layers[i](x)
            vel = self.vel_layers[i](vel)
            bbox2vel = self.cross_bbox2vel[i](x, vel)
            vel2bbox = self.cross_vel2bbox[i](vel, x)
            vel, x = vel2bbox, bbox2vel

        return x, vel#[batch_size, seq_len, d_model]


class Model(nn.Module):
    def __init__(self, num_layers, d_model, bbox_input, speed_input, num_heads, dff, maximum_position_encoding, rate=0.1):
        super(Model, self).__init__()

        self.encoder = Encoder(num_layers, d_model, bbox_input, speed_input, num_heads, dff, maximum_position_encoding, rate)

        self.resize = nn.Linear(d_model * 2, d_model)

        self.att_t = Time_att(d_model)

        self.linear = nn.Linear(d_model, 4)
        self.act1 = nn.ReLU()
        self.dense = nn.Linear(4, 1)
        self.activation = nn.Sigmoid()

    def forward(self, data):
        x = data[:, :, :-2]
        
        vel = data[:, :, -2:]
        x, vel = self.encoder(x, vel)
        y = torch.cat((x, vel), dim=-1)# [32, 16, 256]
        #y = torch.mean(self.resize(y), dim=1)
        y = self.att_t(self.resize(y))

        y = self.act1(self.linear(y))
        return self.activation(self.dense(y))



class Time_att(nn.Module):
    def __init__(self, dim):
        super(Time_att, self).__init__()

        self.linear1 = nn.Linear(dim, dim, bias=False)
        self.linear2 = nn.Linear(dim, 1, bias=False)


    def forward(self, x):
        x = x.contiguous()
        y = self.linear2(torch.tanh(self.linear1(x)))# [B, N, 1]
        
        beta = F.softmax(y, dim=1) # [B,N,1]

        c = beta * x#[B, N, 1] * [B, N, C] = [B, N, C]
        c = torch.sum(c, dim=1).squeeze()# [N, C]
        return c
    


'''model = Model(num_layers=4, d_model=128, bbox_input=4, speed_input=2, num_heads=8, dff=256, maximum_position_encoding=16)

data = torch.randn(size=(32, 16, 6))

y = model(data)
print(y.shape)'''

'''model2 = Time_att(256)
tensor = torch.randn(size=(32, 16, 256))
res = model2(tensor)
print(res.shape)#expected res's size -> [32, 256]'''
