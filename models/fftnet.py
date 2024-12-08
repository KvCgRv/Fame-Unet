

# class AFFAttentionBlock(nn.Module):  
#     def __init__(self, in_channels, out_channels, F_g, F_l, F_int):  
#         super(AFFAttentionBlock, self).__init__() 

#         self.hidden_size = 512
#                 # 初始化权重和偏置  
#         self.w1 = (nn.Parameter(torch.randn(in_channels, self.hidden_size, out_channels)),  
#                    nn.Parameter(torch.randn(in_channels, self.hidden_size, out_channels)))  # 512是outputs的通道数  
#         self.w2 = (nn.Parameter(torch.randn(in_channels, self.hidden_size, out_channels)),  # 假设下一层输出也是512通道  
#                    nn.Parameter(torch.randn(in_channels, self.hidden_size, out_channels)))  # 另一个权重矩阵  

#         self.b1 = nn.Parameter(torch.randn(2, 1, out_channels, 1, 1))  
#         self.b2 = nn.Parameter(torch.randn(2, 1, out_channels, 1, 1))  
         
        
#         # MBConv模块  
#         self.mbconv = nn.Sequential(  
#             nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),  
#             nn.BatchNorm2d(out_channels),  
#             nn.ReLU(inplace=True),  
#             nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),  
#             nn.BatchNorm2d(out_channels),  
#             nn.ReLU(inplace=True)  
#         )  

#         # 1x1卷积用于通道混合  
#         self.group_linear = nn.Conv2d(out_channels, out_channels, kernel_size=1)  

#         # 注意力模块的设置  
#         self.W_g = nn.Sequential(  
#             nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),  
#             nn.BatchNorm2d(F_int)  
#         )  

#         self.W_x = nn.Sequential(  
#             nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),  
#             nn.BatchNorm2d(F_int)  
#         )  

#         self.psi = nn.Sequential(  
#             nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),  
#             nn.BatchNorm2d(1),  
#             nn.Sigmoid()  
#         )  


#     def mask(self, x):
#         bias = x

#         dtype = x.dtype
#         x = x.float()

#         x = torch.fft.fft2(x, dim=(2, 3), norm="ortho")
#         # print(f"x shape: {x.shape}")
#         origin_ffted = x#傅里叶原始x张量
#         B, C, H, W = x.shape
#         device = x.device  
#         self.w1 = [w.to(device) for w in self.w1]  
#         self.w2 = [w.to(device) for w in self.w2]  
#         self.b1.to(device)
#         self.b2.to(device) 
#         # print(f"======================x.device{x.device}===============w1.device{self.w1[0].device}")

#         o1_real = F.relu(
#             torch.einsum('bchw,cio->bohw', x.real, self.w1[0]) - \
#             torch.einsum('bchw,cio->bohw', x.imag, self.w1[1]) + \
#             self.b1[0]
#         )
#         o1_imag = F.relu(
#             torch.einsum('bchw,cio->bohw', x.imag, self.w1[0]) + \
#             torch.einsum('bchw,cio->bohw', x.real, self.w1[1]) + \
#             self.b1[1]
#         )

#         o2_real = (
#                 torch.einsum('bchw,cio->bohw', o1_real, self.w2[0]) - \
#                 torch.einsum('bchw,cio->bohw', o1_imag, self.w2[1]) + \
#                 self.b2[0]
#         )

#         o2_imag = (
#                 torch.einsum('bchw,cio->bohw', o1_imag, self.w2[0]) + \
#                 torch.einsum('bchw,cio->bohw', o1_real, self.w2[1]) + \
#                 self.b2[1]
#                     )
#         x_real = F.softshrink(o2_real, lambd=0.001)
#         x_imag = F.softshrink(o2_imag, lambd=0.001)
#         x_mask = torch.stack([x_real, x_imag], dim=-1)
#         x_mask = torch.complex(x_mask[:, :, :, :, 0], x_mask[:, :, :, :, 1])
#         x = x_mask * origin_ffted
#         x = torch.fft.ifft2(x, s=(H, W), dim=(2, 3), norm="ortho")
#         x = x.type(dtype)

#         '''
#         # print(f"o2_real shape: {o2_real.shape}")
#         # print(f"o2_imag shape: {o2_imag.shape}")
            
#         # x = torch.complex(o2_real, o2_imag)
#         x = torch.stack([o2_real, o2_imag], dim=-1) 
#         # print(f"x curr shape: {x.shape}")
#         x = torch.complex(x[:, :, :, :, 0], x[:, :, :, :, 1])
#         x_mask = F.softshrink(x, lambd=0.001)
#         # print(f"x shape: {x.shape}")
#         x = x_mask * origin_ffted
#         x = torch.fft.ifft2(x, s=(H, W), dim=(2, 3), norm="ortho")
#         # print(f"x shape: {x.shape}")
#         # print(f"bias shape: {bias.shape}")
#         x = x.type(dtype) '''

#         return x + bias

#     def forward(self, g, x): 
#         x_filtered = self.mask(x) #获得的x_filtered就是x + x_ffted

#         g1 = self.W_g(g) 
#         x1= self.W_x(x) 
        
#         psi = F.relu(g1 + x1) 
#         psi = self.psi(psi)

#         return x_filtered + g  + (x+g) * psi


import torch  
import torch.nn as nn  
import torch.nn.functional as F  
from torch.nn import init  

class conv_block(nn.Module):
 
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in,
                      ch_out,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=True), 
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
 
            nn.Conv2d(ch_out,
                      ch_out,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=True), 
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True))
 
    def forward(self, x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out, convTranspose=True):
        super(up_conv, self).__init__()
        if convTranspose:
            self.up = nn.ConvTranspose2d(in_channels=ch_in, out_channels=ch_in,kernel_size=4,stride=2, padding=1)
        else:
            self.up = nn.Upsample(scale_factor=2)
 
        self.Conv = nn.Sequential(
            nn.Conv2d(ch_in,
                      ch_out,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=True), 
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True))
 
    def forward(self, x):
        x = self.up(x)
        x = self.Conv(x)
        return x
 
class single_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(single_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in,
                      ch_out,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=True), 
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True))
 
    def forward(self, x):
        x = self.conv(x)
        return x

class AFFAttentionBlock(nn.Module):  
    def __init__(self, in_channels, out_channels, F_g, F_l, F_int):  
        super(AFFAttentionBlock, self).__init__()  

        self.hidden_size = 256  

        # Initialize weights and biases  
        self.w1 = nn.Parameter(torch.randn(in_channels, self.hidden_size, out_channels)).cuda() 
            # nn.Parameter(torch.randn(in_channels, self.hidden_size, out_channels)).cuda()  
        self.w2 = nn.Parameter(torch.randn(in_channels, self.hidden_size, out_channels)).cuda()
            # nn.Parameter(torch.randn(in_channels, self.hidden_size, out_channels)).cuda()  


        self.b1 = nn.Parameter(torch.zeros(2, 1, out_channels, 1, 1)).cuda()  
        self.b2 = nn.Parameter(torch.zeros(2, 1, out_channels, 1, 1)).cuda()  

        # MBConv module  
        self.mbconv = nn.Sequential(  
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1).cuda(),  
            nn.BatchNorm2d(out_channels).cuda(),  
            nn.ReLU(inplace=True).cuda(),  
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1).cuda(),  
            nn.BatchNorm2d(out_channels).cuda(),  
            nn.ReLU(inplace=True).cuda() 
        ) 

        # 1x1 convolution for channel mixing  
        self.group_linear = nn.Conv2d(out_channels, out_channels, kernel_size=1).cuda()  

        # Attention module  
        self.W_g = nn.Sequential(  
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True).cuda(),  
            nn.BatchNorm2d(F_int).cuda()  
        )  

        self.W_x = nn.Sequential(  
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True).cuda(),  
            nn.BatchNorm2d(F_int).cuda()  
        )  

        self.psi = nn.Sequential(  
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True).cuda(),  
            nn.BatchNorm2d(1).cuda(),  
            nn.Sigmoid().cuda()  
        )  

    # def mask(self, x):  #原始mask
    #     bias = x  
    #     dtype = x.dtype  

    #     # Convert to floating point for FFT  
    #     x = x.float()  

    #     # Apply FFT  
    #     x_fft = torch.fft.fft2(x, dim=(-2, -1), norm="ortho")  

    #     # Real and imaginary parts processing  
    #     o1_real = (  
    #         torch.einsum('bchw,cio->bohw', x_fft.real, self.w1[0]) -  
    #         torch.einsum('bchw,cio->bohw', x_fft.imag, self.w1[1]) +  
    #         self.b1[0]  
    #     )  
    #     o1_imag = (  
    #         torch.einsum('bchw,cio->bohw', x_fft.imag, self.w1[0]) +  
    #         torch.einsum('bchw,cio->bohw', x_fft.real, self.w1[1]) +  
    #         self.b1[1]  
    #     )  

    #     o2_real = (  
    #         torch.einsum('bchw,cio->bohw', o1_real, self.w2[0]) -  
    #         torch.einsum('bchw,cio->bohw', o1_imag, self.w2[1]) +  
    #         self.b2[0]  
    #     )  

    #     o2_imag = (  
    #         torch.einsum('bchw,cio->bohw', o1_imag, self.w2[0]) +  
    #         torch.einsum('bchw,cio->bohw', o1_real, self.w2[1]) +  
    #         self.b2[1]  
    #     )  

    #     # o2_real = F.softshrink(o2_real, lambd=0.1)  
    #     # o2_imag = F.softshrink(o2_imag, lambd=0.1)  
    #     x_mask = torch.complex(o2_real, o2_imag)
    #     x_mask = F.relu(self.mbconv(torch.fft.ifft2(x_mask,norm="ortho").real))  

    #     # Apply the mask to the original FFT result  
    #     x = x_mask * x_fft  

    #     # Inverse FFT to go back to the spatial domain  
    #     x = torch.fft.ifft2(x, s=(x.size(-2), x.size(-1)), dim=(-2, -1), norm="ortho").real  

    #     # Restore original dtype  
    #     x = x.type(dtype)  

    #     return x + bias  
    def mask(self, x):  
        bias = x  
        dtype = x.dtype  

        # Convert to floating point for FFT  
        x = x.float()  

        # Apply FFT  
        x_fft = torch.fft.fft2(x, dim=(-2, -1), norm="ortho")  

        # Real and imaginary parts processing  
        o1_real = (  
            torch.einsum('bchw,cio->bohw', x_fft.real, self.w1) +  
  
            self.b1[0]  
        )  
        o1_imag = (  
            torch.einsum('bchw,cio->bohw', x_fft.imag, self.w2) +    
            self.b1[1]  
        )  

        o2_real = (  
            torch.einsum('bchw,cio->bohw', o1_real, self.w1) +    
            self.b2[0]  
        )  

        o2_imag = (  
            torch.einsum('bchw,cio->bohw', o1_imag, self.w2) +    
            self.b2[1]  
        )  
        



        x_mask = torch.complex(o2_real, o2_imag)
        x_mask = F.relu(self.mbconv(torch.fft.ifft2(x_mask,norm="ortho").real))  

        # Apply the mask to the original FFT result  
        x = x_mask * x_fft  

        # Inverse FFT to go back to the spatial domain  
        x = torch.fft.ifft2(x, s=(x.size(-2), x.size(-1)), dim=(-2, -1), norm="ortho").real  

        # Restore original dtype  
        x = x.type(dtype)  

        return x + bias  

    def forward(self, g, x):  
        # Apply mask which includes FFT processing  
        x_filtered = self.mask(x)  

        # Compute attention weights  
        g1 = self.W_g(g)  
        #x1 = self.W_x(x)  
        x1 = self.W_x(x_filtered)
        psi = self.psi(F.relu(g1 + x1))  

        # Compute final output with attention mechanism  
        out = x_filtered * psi + g * (1 - psi)  
        return out  










# 修改后的 Attention U-Net  
class FFTU_Net(nn.Module):  
    def __init__(self, img_ch=3, output_ch=1, channel_list=[64, 128, 256, 512, 1024]):  
        super(FFTU_Net, self).__init__()  

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)  

        self.Conv1 = conv_block(ch_in=img_ch, ch_out=channel_list[0])  
        self.Conv2 = conv_block(ch_in=channel_list[0], ch_out=channel_list[1])  
        self.Conv3 = conv_block(ch_in=channel_list[1], ch_out=channel_list[2])  
        self.Conv4 = conv_block(ch_in=channel_list[2], ch_out=channel_list[3])  
        self.Conv5 = conv_block(ch_in=channel_list[3], ch_out=channel_list[4])  

        self.Up5 = up_conv(ch_in=channel_list[4], ch_out=channel_list[3]) 
        self.AFF5 = AFFAttentionBlock(in_channels=channel_list[3], out_channels=channel_list[3], F_g=channel_list[3], F_l=channel_list[3], F_int=channel_list[2]) 
        self.Up_conv5 = conv_block(ch_in=channel_list[4], ch_out=channel_list[3])  

        self.Up4 = up_conv(ch_in=channel_list[3], ch_out=channel_list[2])  
        self.AFF4 = AFFAttentionBlock(in_channels=channel_list[2], out_channels=channel_list[2], F_g=channel_list[2], F_l=channel_list[2], F_int=channel_list[1])
        self.Up_conv4 = conv_block(ch_in=channel_list[3], ch_out=channel_list[2])  

        self.Up3 = up_conv(ch_in=channel_list[2], ch_out=channel_list[1])  
        self.AFF3 = AFFAttentionBlock(in_channels=channel_list[1], out_channels=channel_list[1], F_g=channel_list[1], F_l=channel_list[1], F_int=64)
        self.Up_conv3 = conv_block(ch_in=channel_list[2], ch_out=channel_list[1])  

        self.Up2 = up_conv(ch_in=channel_list[1], ch_out=channel_list[0])  
        self.AFF2 = AFFAttentionBlock(in_channels=channel_list[0], out_channels=channel_list[0], F_g=channel_list[0], F_l=channel_list[0], F_int=channel_list[0] // 2)
        self.Up_conv2 = conv_block(ch_in=channel_list[1], ch_out=channel_list[0])  

        self.Conv_1x1 = nn.Conv2d(channel_list[0], output_ch, kernel_size=1, stride=1, padding=0)  


    def forward(self, x):  
        # encoder  
        x1 = self.Conv1(x)  
        # print(f"x1 shape: {x1.shape}")  
        x2 = self.Maxpool(x1)  
        x2 = self.Conv2(x2)  
        # print(f"x2 shape: {x2.shape}")  
        x3 = self.Maxpool(x2)  
        x3 = self.Conv3(x3)  
        # print(f"x3 shape: {x3.shape}")  
        x4 = self.Maxpool(x3)  
        x4 = self.Conv4(x4)  
        # print(f"x4 shape: {x4.shape}")  
        x5 = self.Maxpool(x4)  
        x5 = self.Conv5(x5)  
        # print(f"x5 shape: {x5.shape}")  

        # decoder  
        d5 = self.Up5(x5)  
        # print(f"d5 shape after Up5: {d5.shape}")
        x4_att = self.AFF5(g=d5, x=x4)  
        d5 = torch.cat((x4_att, d5), dim=1)  
        # print(f"d5 shape after concatenation: {d5.shape}")  
        d5 = self.Up_conv5(d5)  

        d4 = self.Up4(d5)  
        # print(f"d4 shape after Up4: {d4.shape}")
        x3_att = self.AFF4(g=d4, x=x3)
        d4 = torch.cat((x3_att, d4), dim=1)  
        d4 = self.Up_conv4(d4)  

        d3 = self.Up3(d4)  
        # print(f"d3 shape after Up3: {d3.shape}")  
        x2_att = self.AFF3(g=d3, x=x2)  
        d3 = torch.cat((x2_att, d3), dim=1)  
        d3 = self.Up_conv3(d3)  

        d2 = self.Up2(d3)  
        # print(f"d2 shape after Up2: {d2.shape}")  
        x1_att = self.AFF2(g=d2, x=x1)  
        d2 = torch.cat((x1_att, d2), dim=1)  
        d2 = self.Up_conv2(d2)  

        d1 = self.Conv_1x1(d2)  

        return {"out": d1}
