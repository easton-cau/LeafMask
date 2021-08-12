import torch
import torch.nn as nn

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


# 1.Global Channel Attention
class GlobalChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(GlobalChannelAttention, self).__init__()

        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.global_att_conv = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_planes // ratio),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_planes)
        )
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        global_max_out = self.max_pool(x)
        global_max_out = self.global_att_conv(global_max_out)
        global_avg_out = self.avg_pool(x)
        global_avg_out = self.global_att_conv(global_avg_out)

        out = global_max_out + global_avg_out
        return self.sigmoid(out)


# 2.Global Local Channel Attention
class GlobalLocalChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(GlobalLocalChannelAttention, self).__init__()

        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.global_att_conv = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_planes // ratio),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_planes)
        )
        self.local_att = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_planes // ratio),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_planes)
        )
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        global_max_out = self.max_pool(x)
        global_max_out = self.global_att_conv(global_max_out)
        global_avg_out = self.avg_pool(x)
        global_avg_out = self.global_att_conv(global_avg_out)
        local_out = self.local_att(x)

        out = global_max_out + global_avg_out + local_out
        return self.sigmoid(out)


# 3.Global Local Spatial Attention
class GlobalLocalSpatialAttention(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(GlobalLocalSpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.global_conv = nn.Sequential(
            nn.Conv2d(2, in_planes // ratio, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(in_planes // ratio),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_planes)
        )
        self.local_att = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_planes // ratio),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_planes)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        global_max_out, _ = torch.max(x, dim=1, keepdim=True)
        global_avg_out = torch.mean(x, dim=1, keepdim=True)
        global_out = torch.cat([global_max_out, global_avg_out], dim=1)
        global_out = self.global_conv(global_out)
        local_out = self.local_att(x)

        out = global_out + local_out
        return self.sigmoid(out)


# 4.Global Local Dual Attention Parallel
class GlobalLocalDualAttention_P(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(GlobalLocalDualAttention_P, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.global_conv = nn.Sequential(
            nn.Conv2d(2, in_planes // ratio, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(in_planes // ratio),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_planes)
        )
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.global_att_conv = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_planes // ratio),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_planes)
        )

        self.local_att_s = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_planes // ratio),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_planes)
        )
        self.local_att_c = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_planes // ratio),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_planes)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        global_max_out, _ = torch.max(x, dim=1, keepdim=True)
        global_avg_out = torch.mean(x, dim=1, keepdim=True)
        global_out = torch.cat([global_max_out, global_avg_out], dim=1)
        global_out = self.global_conv(global_out)
        local_out_s = self.local_att_s(x)
        out_s = global_out + local_out_s

        global_max_out = self.max_pool(x)
        global_max_out = self.global_att_conv(global_max_out)
        global_avg_out = self.avg_pool(x)
        global_avg_out = self.global_att_conv(global_avg_out)
        local_out_c = self.local_att_c(x)
        out_c = global_max_out + global_avg_out + local_out_c

        out = out_s + out_c
        return self.sigmoid(out)


# 5.Global Local Dual Attention Parallel Shared
class GlobalLocalDualAttention_PS(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(GlobalLocalDualAttention_PS, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.global_conv = nn.Sequential(
            nn.Conv2d(2, in_planes // ratio, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(in_planes // ratio),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_planes)
        )
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.global_att_conv = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_planes // ratio),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_planes)
        )

        self.local_att = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_planes // ratio),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_planes)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        global_max_out, _ = torch.max(x, dim=1, keepdim=True)
        global_avg_out = torch.mean(x, dim=1, keepdim=True)
        global_out = torch.cat([global_max_out, global_avg_out], dim=1)
        out_s = self.global_conv(global_out)

        global_max_out = self.max_pool(x)
        global_max_out = self.global_att_conv(global_max_out)
        global_avg_out = self.avg_pool(x)
        global_avg_out = self.global_att_conv(global_avg_out)
        out_c = global_max_out + global_avg_out

        local_out = self.local_att(x)

        out = out_s + out_c + local_out
        return self.sigmoid(out)