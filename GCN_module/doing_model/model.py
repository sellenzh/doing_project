import math

from torch import nn
import numpy as np

from model_0810.TCN_GCN_unit import TCN_GCN_unit
from model_0810.TCN_GCN_unit import bn_init
from model_0810.gated_fusion import CrossTransformer
from model_0810.gated_fusion import GatedFusion


class pedMondel(nn.Module):
    def __init__(self, frames, vel=False, seg=False, h3d=True, nodes=19, n_clss=1):
        super(pedMondel, self).__init__()
        self.n_clss = n_clss
        self.ch = 4
        self.ch1, self.ch2 = 32, 64
        self.width, self.height = 192, 64
        self.time, self.nodes = 62, nodes

        self.data_bn = nn.BatchNorm1d(self.ch * nodes)
        bn_init(self.data_bn, 1)
        self.drop = nn.Dropout(0.25)
        A = np.stack([np.eye(nodes)] * 3, axis=0)

        self.img1 = nn.Sequential(
                nn.Conv2d(self.ch, self.ch1, kernel_size=3, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(self.ch1), nn.SiLU())

        self.vel1 = nn.Sequential(
                nn.Conv1d(2, self.ch1, 3, bias=False), nn.BatchNorm1d(self.ch1), nn.SiLU())
        # ----------------------------------------------------------------------------------------------------
        self.layer1 = TCN_GCN_unit(self.ch, self.ch1, A, residual=False)
        
        '''self.cross_img1 = CrossTransformer(inputs=32)
        self.cross_vel1 = CrossTransformer(inputs=32)

        self.gated_img1 = GatedFusion(dims=32)
        self.gated_vel1 = GatedFusion(dims=32)'''

        self.img2 = nn.Sequential(
                nn.Conv2d(self.ch1, self.ch1, kernel_size=3, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(self.ch1), nn.SiLU())

        self.vel2 = nn.Sequential(
                nn.Conv1d(self.ch1, self.ch1, 3, bias=False),
                nn.BatchNorm1d(self.ch1), nn.SiLU())
        # ----------------------------------------------------------------------------------------------------
        self.layer2 = TCN_GCN_unit(self.ch1, self.ch2, A)

        '''self.cross_img2 = CrossTransformer(inputs=64)
        self.cross_vel2 = CrossTransformer(inputs=64)
        
        self.gated_img2 = GatedFusion(dims=64)
        self.gated_vel2 = GatedFusion(dims=64)'''

        self.img3 = nn.Sequential(
                nn.Conv2d(self.ch1, self.ch2, kernel_size=2, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(self.ch2), nn.SiLU())

        self.vel3 = nn.Sequential(
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

        self.pool_sig_2d = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Sigmoid())
        
        self.pool_sig_1d = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Sigmoid())

        # ----------------------------------------------------------------------------------------------------
        '''self.img_linear_time = nn.Sequential(
            nn.Linear(self.width, self.time * 3, bias=False),
            nn.BatchNorm2d(self.ch1), nn.ReLU(),
            nn.Linear(self.time * 3, self.time * 3, bias=False),
            nn.BatchNorm2d(self.ch1), nn.ReLU(),
            nn.Linear(self.time * 3, self.time, bias=False),
            nn.BatchNorm2d(self.ch1), nn.ReLU())

        self.img_linear_nodes = nn.Sequential(
            nn.Linear(self.height, self.nodes * 3, bias=False),
            nn.BatchNorm2d(self.ch1), nn.ReLU(),
            nn.Linear(self.nodes * 3, self.nodes * 3, bias=False),
            nn.BatchNorm2d(self.ch1), nn.ReLU(),
            nn.Linear(self.nodes * 3, self.nodes, bias=False),
            nn.BatchNorm2d(self.ch1), nn.ReLU())

        # ----------------------------------------------------------------------------------------------------
        self.img_channel1 = nn.Sequential(
            nn.Conv2d(self.ch, self.ch1 * 2, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.ch1 * 2), nn.SiLU(),
            nn.Conv2d(self.ch1 * 2, self.ch1 * 2, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.ch1 * 2), nn.SiLU(),
            nn.Conv2d(self.ch1 * 2, self.ch1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.ch1), nn.SiLU(),
            nn.BatchNorm2d(self.ch1), nn.ReLU())

        self.img_channel2 = nn.Sequential(
            nn.Conv2d(self.ch1, self.ch2 * 2, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.ch2 * 2), nn.SiLU(),
            nn.Conv2d(self.ch2 * 2, self.ch2 * 2, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.ch2 * 2), nn.SiLU(),
            nn.Conv2d(self.ch2 * 2, self.ch2, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.ch2), nn.SiLU(),
            nn.BatchNorm2d(self.ch2), nn.ReLU())

        # ----------------------------------------------------------------------------------------------------
        self.vel_nodes = nn.Sequential(
            nn.Linear(1, self.nodes * 3, bias=False),
            nn.BatchNorm2d(self.ch1), nn.ReLU(),
            nn.Linear(self.nodes * 3, self.nodes * 3, bias=False),
            nn.BatchNorm2d(self.ch1), nn.ReLU(),
            nn.Linear(self.nodes * 3, self.nodes, bias=False),
            nn.BatchNorm2d(self.ch1), nn.ReLU())

        self.vel_channel1 = nn.Sequential(
            nn.Conv1d(2, self.ch1 * 2, kernel_size=1, bias=False),
            nn.BatchNorm1d(self.ch1 * 2), nn.SiLU(),
            nn.Conv1d(self.ch1 * 2, self.ch1 * 2, kernel_size=1, bias=False),
            nn.BatchNorm1d(self.ch1 * 2), nn.SiLU(),
            nn.Conv1d(self.ch1 * 2, self.ch1, kernel_size=1, bias=False),
            nn.BatchNorm1d(self.ch1), nn.SiLU(),
            nn.BatchNorm1d(self.ch1), nn.ReLU())

        self.vel_channel2 = nn.Sequential(
            nn.Conv2d(self.ch1, self.ch2 * 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.ch2 * 2), nn.SiLU(),
            nn.Conv2d(self.ch2 * 2, self.ch2 * 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.ch2 * 2), nn.SiLU(),
            nn.Conv2d(self.ch2 * 2, self.ch2, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.ch2), nn.SiLU(),
            nn.BatchNorm2d(self.ch2), nn.ReLU())'''
        # ----------------------------------------------------------------------------------------------------

    def forward(self, kp, frame=None, vel=None):

        N, C, T, V = kp.shape
        kp = kp.permute(0, 1, 3, 2).contiguous().view(N, C * V, T)
        kp = self.data_bn(kp)
        kp = kp.view(N, C, V, T).permute(0, 1, 3, 2).contiguous()  # [2, 4, T, 19]

        f1 = self.img1(frame)  # [2, 32, 190, 62]
        v1 = self.vel1(vel)  # [2, 32, T-2]

        pose = self.layer1(kp)
        f1 = self.img2(f1)
        pose.mul(self.pool_sig_2d(f1))
        v1 = self.vel2(v1)
        pose = pose.mul(self.pool_sig_1d(v1).unsqueeze(-1))
        
        '''fr = self.img_channel1(frame)
        fr = self.img_linear_time(fr.permute(0, 1, 3, 2))
        fr = self.img_linear_nodes(fr.permute(0, 1, 3, 2))
        pose_att1 = self.cross_img1(pose, fr[:, :, -T:, :])
        pose_img = self.gated_img1(pose, pose_att1)

        ve = self.vel_channel1(vel[:, :, -T:])
        ve = self.vel_nodes(ve.unsqueeze(-1))
        pose_att2 = self.cross_vel1(pose, ve)
        pose = self.gated_vel1(pose_img, pose_att2)'''

        pose = self.layer2(pose)
        f1 = self.img3(f1)
        pose = pose.mul(self.pool_sig_2d(f1))
        v1 = self.vel3(v1)
        pose = pose.mul(self.pool_sig_1d(v1).unsqueeze(-1))

        '''fr = self.img_channel2(fr)
        pose_att1 = self.cross_img2(pose, fr[:, :, -T:, :])
        pose_img = self.gated_img2(pose, pose_att1)

        ve = self.vel_channel2(ve)
        pose_att2 = self.cross_vel2(pose_img, ve)
        pose = self.gated_vel2(pose_img, pose_att2)'''

        pose = self.gap(pose).squeeze(-1)
        pose = pose.squeeze(-1)
        y = self.att(pose).mul(pose) + pose
        y = self.drop(y)
        y = self.linear(y)

        return y
