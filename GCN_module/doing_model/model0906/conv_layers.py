from torch import nn

class ConvAndPoolImg(nn.Module):
    def __init__(self, ch_in, ch_1st, ch_2nd):
        super(ConvAndPoolImg, self).__init__()
        self.ch, self.ch1, self.ch2 = ch_in, ch_1st, ch_2nd
        self.img_w, self.img_h = 192, 64
        self.nodes = 19
        self.time = 62

        self.conv1 = nn.Sequential(
            nn.Conv2d(self.ch, self.ch1, kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.ch1), nn.SiLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(self.ch1, self.ch1, kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.ch1), nn.SiLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(self.ch1, self.ch2, kernel_size=2, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.ch2), nn.SiLU()
        )
        self.pool_sig = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Sigmoid()
        )
        #--------------------------------------------------------------------------------
        self.linear1 = nn.Sequential(
            nn.Conv2d(self.ch, self.ch1 * 2, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.ch1 * 2), nn.ReLU(),
            nn.Conv2d(self.ch1 * 2, self.ch1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.ch1), nn.ReLU(),
            nn.BatchNorm2d(self.ch1), nn.ReLU()
        )
        self.linear2 = nn.Sequential(
            nn.Conv2d(self.ch1, self.ch2 * 2, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.ch2 * 2), nn.ReLU(),
            nn.Conv2d(self.ch2 * 2, self.ch2, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.ch2), nn.ReLU(),
            nn.BatchNorm2d(self.ch2), nn.ReLU()
        )

        self.linear_nodes = nn.Sequential(
            nn.Linear(self.img_h, self.nodes * 3, bias=False),
            #nn.LayerNorm(self.nodes * 3), nn.SiLU(),
            nn.BatchNorm2d(self.ch1), nn.ReLU(),
            nn.Linear(self.nodes * 3, self. nodes * 3, bias=False),
            #nn.LayerNorm(self.nodes * 3), nn.SiLU(),
            nn.BatchNorm2d(self.ch1), nn.ReLU(),
            nn.Linear(self.nodes * 3, self.nodes, bias=False),
            #nn.LayerNorm(self.nodes), nn.SiLU(),
            nn.BatchNorm2d(self.ch1), nn.ReLU(),
            nn.BatchNorm2d(self.ch1), nn.ReLU()
        )
        self.linear_time = nn.Sequential(
            nn.Linear(self.img_w, self.time * 3, bias=False),
            #nn.LayerNorm(self.time * 3), nn.SiLU(),
            nn.BatchNorm2d(self.ch1), nn.ReLU(),
            nn.Linear(self.time * 3, self.time * 3, bias=False),
            #nn.LayerNorm(self.time * 3), nn.SiLU(),
            nn.BatchNorm2d(self.ch1), nn.ReLU(),
            nn.Linear(self.time * 3, self.time, bias=False),
            #nn.LayerNorm(self.time), nn.SiLU(),
            nn.BatchNorm2d(self.ch1), nn.ReLU(),
            nn.BatchNorm2d(self.ch1), nn.ReLU()
        )

    def forward(self, img, time=None):
        frame = self.conv1(img)
        frame = self.conv2(frame)
        img1 = self.pool_sig(frame)
        img2  = self.pool_sig(self.conv3(frame))

        fr = self.linear1(img)
        fr = self.linear_nodes(fr)
        fr1 = self.linear_time(fr.permute(0, 1, 3, 2)).permute(0, 1, 3, 2).contiguous()
        if time is not None:
            fr1 = fr1[:, :, -time:, :]
        fr2 = self.linear2(fr1)

        return img1, img2, fr1, fr2

class ConvAndPoolVel(nn.Module):
    def __init__(self, ch_in, ch_1st, ch_2nd):
        super(ConvAndPoolVel, self).__init__()
        self.ch, self.ch1, self.ch2 = ch_in, ch_1st, ch_2nd
        self.nodes = 19
        self.time = 62

        self.vel1 = nn.Sequential(
            nn.Conv1d(2, self.ch1, 3, bias=False), nn.BatchNorm1d(self.ch1), nn.SiLU()
        )
        self.vel2 = nn.Sequential(
            nn.Conv1d(self.ch1, self.ch1, 3, bias=False),
            nn.BatchNorm1d(self.ch1), nn.SiLU()
        )
        self.vel3 = nn.Sequential(
            nn.Conv1d(self.ch1, self.ch2, kernel_size=2, bias=False),
            nn.BatchNorm1d(self.ch2), nn.SiLU()
        )
        self.pool_sig = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Sigmoid()
        )

        self.vel_nodes = nn.Sequential(
            nn.Linear(1, self.nodes * 3, bias=False),
            #nn.LayerNorm(self.nodes * 3), nn.SiLU(),
            nn.BatchNorm2d(self.ch1), nn.ReLU(),
            nn.Linear(self.nodes * 3, self.nodes * 3, bias=False),
            #nn.LayerNorm(self.nodes * 3), nn.SiLU(),
            nn.BatchNorm2d(self.ch1), nn.ReLU(),
            nn.Linear(self.nodes * 3, self.nodes, bias=False),
            #nn.LayerNorm(self.nodes), nn.SiLU(),
            nn.BatchNorm2d(self.ch1), nn.ReLU(),
            nn.BatchNorm2d(self.ch1), nn.ReLU()
        )
        self.linear1 = nn.Sequential(
            nn.Conv1d(2, self.ch1 * 2, kernel_size=1, bias=False),
            nn.BatchNorm1d(self.ch1 * 2), nn.ReLU(),
            nn.Conv1d(self.ch1 * 2, self.ch1 * 2, kernel_size=1, bias=False),
            nn.BatchNorm1d(self.ch1 * 2), nn.ReLU(),
            nn.Conv1d(self.ch1 * 2, self.ch1, kernel_size=1, bias=False),
            nn.BatchNorm1d(self.ch1), nn.ReLU(),
            nn.BatchNorm1d(self.ch1), nn.ReLU()
        )
        self.linear2 = nn.Sequential(
            nn.Conv2d(self.ch1, self.ch2 * 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.ch2 * 2), nn.ReLU(),
            nn.Conv2d(self.ch2 * 2, self.ch2 * 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.ch2 * 2), nn.ReLU(),
            nn.Conv2d(self.ch2 * 2, self.ch2, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.ch2), nn.ReLU(),
            nn.BatchNorm2d(self.ch2), nn.ReLU()
        )
    def forward(self, speed, time=None):
        vel = self.vel1(speed)
        vel = self.vel2(vel)
        vel1 = self.pool_sig(vel).unsqueeze(-1)
        vel2 = self.pool_sig(self.vel3(vel)).unsqueeze(-1)


        if time is not None:
            ve = speed[:, :, -time:]
        ve = self.linear1(ve)
        ve1 = self.vel_nodes(ve.unsqueeze(-1))
        ve2 = self.linear2(ve1)

        return vel1, vel2, ve1, ve2