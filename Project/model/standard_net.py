from torch import nn

from . import common


class PedModel(nn.Module):
    def __init__(self, args, n_clss=1):
        super().__init__()
        self.nodes = args.nodes
        self.n_clss = n_clss
        self.ch, self.ch1, self.ch2 = 4, 32, 64

        self.data_bn = common.DataBN(self.ch, args)

        self.img = common.ImgConvLayers(self.ch, self.ch1, self.ch2)
        self.vel = common.VelConvLayers(self.ch, self.ch1, self.ch2)

        self.layers = common.GCN_TAT_layers(self.ch, self.ch1, self.ch2, args)
        #-----------------------------------------------------------
        self.process = common.Process(self.ch2, self.n_clss)

    def forward(self, kp, frame, vel):
        kp = self.data_bn(kp)

        img = self.img(frame)
        vel1, vel2 = self.vel(vel)

        pose = self.layers(kp, img, vel1, vel2)

        y = self.process(pose)
        return y



def conv_init(conv):
    if conv.weight is not None:
        nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)

def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)