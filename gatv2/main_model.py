from torch import nn
import math

from pose_module import Pose_Module


class Model(nn.Module):
    def __init__(self, args, n_class=1):
        super(Model, self).__init__()
        self.ch, self.ch1, self.ch2 = 4, 32, 64
        self.n_class = n_class
        self.data_bn = nn.BatchNorm1d(4 * 19)
        bn_init(self.data_bn, 1)

        #---------------image convolution layer----------------------
        self.img_layer0 = nn.Sequential(
            nn.Conv2d(self.ch, self.ch1, kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.ch1), nn.SiLU())
        self.img_layer1 = nn.Sequential(
            nn.Conv2d(self.ch1, self.ch1, kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.ch1), nn.SiLU())
        self.img_layer2 = nn.Sequential(
            nn.Conv2d(self.ch1, self.ch2, kernel_size=2, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.ch2), nn.SiLU())
        #---------------speed convolution layer-----------------------
        self.speed_layer0 = nn.Sequential(
            nn.Conv1d(2, self.ch1, kernel_size=3, bias=False),
            nn.BatchNorm1d(self.ch1), nn.SiLU())
        self.speed_layer1 = nn.Sequential(
            nn.Conv1d(self.ch1, self.ch1, kernel_size=3, bias=False),
            nn.BatchNorm1d(self.ch1), nn.SiLU())
        self.speed_layer2 = nn.Sequential(
            nn.Conv1d(self.ch1, self.ch2, kernel_size=2, bias=False),
            nn.BatchNorm1d(self.ch2), nn.SiLU())
        #-----------------bbox convolution layer-----------------------
        self.bbox_layer0 = nn.Sequential(
            nn.Conv1d(4, self.ch1, kernel_size=3, bias=False),
            nn.BatchNorm1d(self.ch1), nn.SiLU())
        self.bbox_layer1 = nn.Sequential(
            nn.Conv1d(self.ch1, self.ch1, kernel_size=3, bias=False),
            nn.BatchNorm1d(self.ch1), nn.SiLU())
        self.bbox_layer2 = nn.Sequential(
            nn.Conv1d(self.ch1, self.ch2, kernel_size=2, bias=False),
            nn.BatchNorm1d(self.ch2), nn.SiLU())
        #-------------pose attention module-----------------------------
        self.pose_layer1 = Pose_Module(config=args, first=True)
        self.pose_layer2 = Pose_Module(config=args, first=False)

        self.pool_sig_2d = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Sigmoid())
        self.pool_sig_1d_s = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Sigmoid())
        self.pool_sig_1d_b = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.SiLU())

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.att = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.ch2, self.ch2, bias=False),
            nn.BatchNorm1d(self.ch2),
            nn.Sigmoid())
        self.linear = nn.Linear(self.ch2, self.n_class)
        nn.init.normal_(self.linear.weight, 0, math.sqrt(2. / self.n_class))

        self.drop = nn.Dropout(0.25)
    

    def forward(self, pose, frames, velocity, bbox):
        N, C, T, V = pose.shape
        pose = pose.permute(0, 1, 3, 2).contiguous().view(N, C*T, V)
        pose = self.data_bn(pose)
        pose = pose.view(N, C, T, V).permute(0, 1, 3, 2).contiguous()

        x = self.pose_layer1(pose)
        fr =self.img_layer1(self.img_layer0(frames))
        vel = self.speed_layer1(self.speed_layer0(velocity))
        bb = self.bbox_layer1(self.bbox_layer0(bbox))
        x.mul(self.pool_sig_2d(fr))
        x.mul(self.pool_sig_1d_s(vel).unsqueeze(-1))
        x.mul(self.pool_sig_1d_b(bb).unsqueeze(-1))

        x = self.pose_layer2(x)
        x.mul(self.pool_sig_2d(self.img_layer2(fr)))
        x.mul(self.pool_sig_1d_s(self.speed_layer2(vel)).unsqueeze(-1))
        x.mul(self.pool_sig_1d_b(self.bbox_layer2(bb)).unsqueeze(-1))

        x = self.gap(x).squeeze(-1)
        x = x.squeeze(-1)
        x = self.att(x).mul(x) + x
        y = self.linear(self.drop(x))   
        return y

def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)

