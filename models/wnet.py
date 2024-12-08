# Loosely inspired on https://github.com/jvanvugt/pytorch-unet
# Improvements (conv_bridge, shortcut) added by A. Galdran (Dec. 2019)

import torch
import torch.nn as nn
import torch.nn.functional as F  

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class ConvBlock(torch.nn.Module):
    def __init__(self, in_c, out_c, k_sz=3, shortcut=False, pool=True):
        '''
        pool_mode can be False (no pooling) or True ('maxpool')
        '''
        super(ConvBlock, self).__init__()
        if shortcut==True: self.shortcut = nn.Sequential(conv1x1(in_c, out_c), nn.BatchNorm2d(out_c))
        else: self.shortcut=False
        pad = (k_sz - 1) // 2

        block = []
        if pool: self.pool = nn.MaxPool2d(kernel_size=2)
        else: self.pool = False

        block.append(nn.Conv2d(in_c, out_c, kernel_size=k_sz, padding=pad))
        block.append(nn.ReLU())
        block.append(nn.BatchNorm2d(out_c))

        block.append(nn.Conv2d(out_c, out_c, kernel_size=k_sz, padding=pad))
        block.append(nn.ReLU())
        block.append(nn.BatchNorm2d(out_c))

        self.block = nn.Sequential(*block)
    def forward(self, x):
        if self.pool: x = self.pool(x)
        out = self.block(x)
        if self.shortcut: return out + self.shortcut(x)
        else: return out

class UpsampleBlock(torch.nn.Module):
    def __init__(self, in_c, out_c, up_mode='transp_conv'):
        super(UpsampleBlock, self).__init__()
        block = []
        if up_mode == 'transp_conv':
            block.append(nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2))
        elif up_mode == 'up_conv':
            block.append(nn.Upsample(mode='bilinear', scale_factor=2, align_corners=False))
            block.append(nn.Conv2d(in_c, out_c, kernel_size=1))
        else:
            raise Exception('Upsampling mode not supported')

        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out

class ConvBridgeBlock(torch.nn.Module):
    def __init__(self, channels, k_sz=3):
        super(ConvBridgeBlock, self).__init__()
        pad = (k_sz - 1) // 2
        block=[]

        block.append(nn.Conv2d(channels, channels, kernel_size=k_sz, padding=pad))
        block.append(nn.ReLU())
        block.append(nn.BatchNorm2d(channels))

        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out

class UpConvBlock(torch.nn.Module):
    def __init__(self, in_c, out_c, k_sz=3, up_mode='up_conv', conv_bridge=False, shortcut=False):
        super(UpConvBlock, self).__init__()
        self.conv_bridge = conv_bridge
        self.out_c = out_c
        self.up_layer = UpsampleBlock(in_c, out_c, up_mode=up_mode)
        self.conv_layer = ConvBlock(2 * out_c, out_c, k_sz=k_sz, shortcut=shortcut, pool=False)
        if self.conv_bridge:
            self.conv_bridge_layer = ConvBridgeBlock(out_c, k_sz=k_sz)
        self.mask_conv = torch.nn.Conv2d(2 * out_c, out_c, kernel_size=1)  

    def option(self, out, up):
        #print(f"up.shape: {up.shape}")
        #print(f"out.shape: {out.shape}")
        mask = F.relu(self.mask_conv(out))
        #print(f"mask.shape: {mask.shape}")
        up_fft = torch.fft.fft2(up, dim=(2, 3), norm="ortho")
        up_filtered = up_fft * mask
        up_ifft = torch.fft.ifft2(up_filtered, dim=(2, 3), norm="ortho") + up

        return up_ifft

    def forward(self, x, skip):
        up = self.up_layer(x)
        if self.conv_bridge:
            "=================================================================================================================================================================="
            #print(f"up.shape: {up.shape}")
            #print(f"self.conv_bridge_layer(skip).shape: {self.conv_bridge_layer(skip).shape}")
            out0 = torch.cat([up, self.conv_bridge_layer(skip)], dim=1)
            out1 = self.option(out0, up)
            out = torch.cat([up, out1], dim=1).real

            #print(f"out.shape: {out.shape}")
            #print(f"out channel{self.out_c//2}")
        else:
            out = torch.cat([up, skip], dim=1)


        out = self.conv_layer(out)
        return out

class UNet(nn.Module):
    def __init__(self, in_c, n_classes, layers, k_sz=3, up_mode='transp_conv', conv_bridge=True, shortcut=True):
        super(UNet, self).__init__()
        self.n_classes = n_classes
        self.first = ConvBlock(in_c=in_c, out_c=layers[0], k_sz=k_sz,
                               shortcut=shortcut, pool=False)

        self.down_path = nn.ModuleList()
        for i in range(len(layers) - 1):
            block = ConvBlock(in_c=layers[i], out_c=layers[i + 1], k_sz=k_sz,
                              shortcut=shortcut, pool=True)
            self.down_path.append(block)

        self.up_path = nn.ModuleList()
        reversed_layers = list(reversed(layers))
        for i in range(len(layers) - 1):
            block = UpConvBlock(in_c=reversed_layers[i], out_c=reversed_layers[i + 1], k_sz=k_sz,
                                up_mode=up_mode, conv_bridge=conv_bridge, shortcut=shortcut)
            self.up_path.append(block)

        # init, shamelessly lifted from torchvision/models/resnet.py
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.final = nn.Conv2d(layers[0], n_classes, kernel_size=1)

    def forward(self, x):
        x = self.first(x)
        down_activations = []
        for i, down in enumerate(self.down_path):
            down_activations.append(x)
            x = down(x)
        down_activations.reverse()
        for i, up in enumerate(self.up_path):
            x = up(x, down_activations[i])
        return self.final(x)

# class WNet(nn.Module):
#     def __init__(self, in_c, n_classes, layers, k_sz=3, up_mode='transp_conv', conv_bridge=True, shortcut=True):
#         super(WNet, self).__init__()
#         self.n_classes = n_classes
#         self.first = ConvBlock(in_c=in_c, out_c=layers[0], k_sz=k_sz,
#                                shortcut=shortcut, pool=False)

#         self.down_path = nn.ModuleList()
#         for i in range(len(layers) - 1):
#             block = ConvBlock(in_c=layers[i], out_c=layers[i + 1], k_sz=k_sz,
#                               shortcut=shortcut, pool=True)
#             self.down_path.append(block)

#         self.up_path = nn.ModuleList()
#         reversed_layers = list(reversed(layers))
#         for i in range(len(layers) - 1):
#             block = UpConvBlock(in_c=reversed_layers[i], out_c=reversed_layers[i + 1], k_sz=k_sz,
#                                 up_mode=up_mode, conv_bridge=conv_bridge, shortcut=shortcut)
#             self.up_path.append(block)

#         self.final = nn.Conv2d(layers[0], n_classes, kernel_size=1)
#         ############################
#         self.first_2 = ConvBlock(in_c=in_c+1, out_c=layers[0], k_sz=k_sz,
#                                  shortcut=shortcut, pool=False)
#         self.down_path_2 = nn.ModuleList()
#         for i in range(len(layers) - 1):
#             block = ConvBlock(in_c=2 * layers[i], out_c=layers[i + 1], k_sz=k_sz,
#                               shortcut=shortcut, pool=True)
#             self.down_path_2.append(block)

#         self.up_path_2 = nn.ModuleList()
#         reversed_layers = list(reversed(layers))
#         for i in range(len(layers) - 1):
#             block = UpConvBlock(in_c=reversed_layers[i], out_c=reversed_layers[i + 1], k_sz=k_sz,
#                                 up_mode=up_mode, conv_bridge=conv_bridge, shortcut=shortcut)
#             self.up_path_2.append(block)
#         self.final_2 = nn.Conv2d(layers[0], n_classes, kernel_size=1)

#         # init, shamelessly lifted from torchvision/models/resnet.py
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#             elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)

#     def forward(self, data):
#         x = self.first(data)
#         down_activations = []
#         up_activations = []

#         for i, down in enumerate(self.down_path):
#             down_activations.append(x)
#             x = down(x)

#         down_activations.reverse()

#         for i, up in enumerate(self.up_path):
#             x = up(x, down_activations[i])
#             up_activations.append(x)

#         out1 = self.final(x)

#         new_data = torch.cat([data, torch.sigmoid(out1)], dim=1)
#         x = self.first_2(new_data)
#         down_activations = []

#         up_activations.reverse()

#         for i, down in enumerate(self.down_path_2):
#             down_activations.append(x)
#             x = down(torch.cat([x, up_activations[i]], dim=1))

#         down_activations.reverse()

#         up_activations = []
#         for i, up in enumerate(self.up_path_2):
#             x = up(x, down_activations[i])
#             up_activations.append(x)
#         out2 = self.final_2(x)

#         return out1, out2

class Attention_block(nn.Module):
 
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True), 
            nn.BatchNorm2d(F_int))
 
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True), 
            nn.BatchNorm2d(F_int))
 
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1), nn.Sigmoid())
 
        self.relu = nn.ReLU(inplace=True)
 
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
 
        return x * psi   

import sys

import torch

# from .res_unet_adrian import WNet as wnet
class W_Net(torch.nn.Module):
    def __init__(self, output_ch=1, in_c=3, layers=(8,16,32), conv_bridge=True, shortcut=True, mode='train'):
        super(W_Net, self).__init__()
        n_classes = output_ch
        self.unet1 = UNet(in_c=in_c, n_classes=n_classes, layers=layers, conv_bridge=conv_bridge, shortcut=shortcut)
        self.unet2 = UNet(in_c=in_c+n_classes, n_classes=n_classes, layers=layers, conv_bridge=conv_bridge, shortcut=shortcut)
        self.n_classes = n_classes
        self.mode=mode

    def forward(self, x):
        x1 = self.unet1(x)
        #print(x1.shape)
        x2 = self.unet2(torch.cat([x, x1], dim=1))
        return {"out":x2}
        


def get_arch(model_name, in_c=3, n_classes=1):

    if model_name == 'unet':
        model = UNet(in_c=in_c, n_classes=n_classes, layers=[8,16,32], conv_bridge=True, shortcut=True)
    elif model_name == 'big_unet':
        model = UNet(in_c=in_c, n_classes=n_classes, layers=[12,24,48], conv_bridge=True, shortcut=True)
    elif model_name == 'wnet':
        model = wnet(in_c=in_c, n_classes=n_classes, layers=[8,16,32], conv_bridge=True, shortcut=True)
    elif model_name == 'big_wnet':
        model = wnet(in_c=in_c, n_classes=n_classes, layers=[8,16,32,64], conv_bridge=True, shortcut=True)


    else: sys.exit('not a valid model_name, check models.get_model.py')

    return model
if __name__ == '__main__':
    import time
    batch_size = 1
    batch = torch.zeros([batch_size, 3, 512, 512], dtype=torch.float32)
    model = get_arch('wnet')
    print("Total params: {0:,}".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    print('Forward pass (bs={:d}) when running in the cpu:'.format(batch_size))
    start_time = time.time()
    logits = model(batch)
    print("--- %s seconds ---" % (time.time() - start_time))