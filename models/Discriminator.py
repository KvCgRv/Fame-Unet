import torch
import torch.nn as nn
#from models.SNConv import SNConv
import numpy as np

def get_pad(in_, ksize, stride, atrous=1):
    out_ = np.ceil(float(in_) / stride)
    return int(((out_ - 1) * stride + atrous * (ksize - 1) + 1 - in_) / 2)


class SNConv(torch.nn.Module):
    """
    SN convolution for spetral normalization conv
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 activation=torch.nn.LeakyReLU(0.2, inplace=True)):
        super(SNConv, self).__init__()
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.conv2d = torch.nn.utils.spectral_norm(self.conv2d)
        self.activation = activation
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, input):
        # print('input', input.shape)
        x = self.conv2d(input)
        # print('output', x.shape)
        if self.activation is not None:
            return self.activation(x)
        else:
            return x
def get_pad(in_, ksize, stride, atrous=1):
    out_ = np.ceil(float(in_) / stride)
    return int(((out_ - 1) * stride + atrous * (ksize - 1) + 1 - in_) / 2)


class Discriminator(nn.Module):
    def __init__(self, inputChannels):
        super(Discriminator, self).__init__()
        cnum = 32
        self.discriminator = nn.Sequential(
            SNConv(inputChannels, 2 * cnum, 5, 1, padding=2),
            SNConv(2 * cnum, 2 * cnum, 3, 2, padding=get_pad(256, 4, 2)),
            SNConv(2 * cnum, 4 * cnum, 3, 2, padding=get_pad(128, 4, 2)),
            SNConv(4 * cnum, 8 * cnum, 3, 2, padding=get_pad(64, 4, 2)),
            SNConv(8 * cnum, 16 * cnum, 3, 2, padding=get_pad(32, 4, 2)),
            SNConv(16 * cnum, 16 * cnum, 3, 2, padding=get_pad(16, 4, 2)),
            SNConv(16 * cnum, 16 * cnum, 3, 2, padding=get_pad(8, 4, 2)),
        )

        # self.shrink = nn.Sequential(
        #     nn.Conv2d(512, 1, kernel_size=8),
        #     nn.Sigmoid()#映射到0,1代表概率
        # )
        self.shrink =  nn.Conv2d(512, 1, kernel_size=8)
        self.sigmoid = nn.Sigmoid()


    def forward(self, input):
        all_feat = self.discriminator(input)
        #print("all_feat.shape",all_feat.shape)#为什么会打印三次1，512，8，8
        all_feat = self.shrink(all_feat)
        all_feat = self.sigmoid(all_feat)
        return all_feat

# test = Discriminator(4)
# ori = torch.ones((1, 3, 256, 256))
# x = torch.ones((1, 1, 256, 256))
# x = test(x, ori)
# print(x.shape)#torch.size[1,1]
# print(x)
