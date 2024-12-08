import torch
import torch.nn as nn
from models.Discriminator import Discriminator
import numpy as np
import torch.nn.functional as F


def dice_loss(input, target):
    input = input.contiguous().view(input.size()[0], -1)
    target = target.contiguous().view(target.size()[0], -1)

    input = input
    target = target

    a = torch.sum(input * target, 1)
    b = torch.sum(input * input, 1) + 0.001
    c = torch.sum(target * target, 1) + 0.001
    d = (2 * a) / (b + c)
    dice_loss = torch.mean(d)
    return 1 - dice_loss


def is_white(img):
    return not np.any(1 - np.array(img.cpu()))


class Loss_Doc(nn.Module):
    def __init__(self, lr=0.00001):
        super(Loss_Doc, self).__init__()
        self.l1 = nn.L1Loss()
        self.cross_entropy = nn.BCELoss()
        self.discriminator_c = Discriminator(6)#三通道真实image，一通道梯度图，一通道阈值图，一通道预测的mask或者真实mask，一共6个通道，可能改成5，去掉了阈值图之后
        self.D_optimizer_c = torch.optim.Adam(self.discriminator_c.parameters(), lr=lr)

    def forward(self,model,x_fake,x_real):
        # l1_loss = self.l1(corase_out, gt) + 2 * self.l1(re_out, gt) + 0.5*(self.l1(
        #     corase_out_ori, gt_ori) + self.l1(corase_out_ori_full, gt_ori_full)) + self.l1(edge_out, gt_Sobel)
        # cross_entropy_loss = self.cross_entropy(corase_out, gt) + 2 * self.cross_entropy(re_out,
        #                                                                                  gt) + 0.5*(self.cross_entropy(corase_out_ori, gt_ori) + self.cross_entropy(corase_out_ori_full, gt_ori_full)) + self.cross_entropy(edge_out, gt_Sobel)
        # if is_white(gt):
        #     flag_white = True
        #     mask_loss = dice_loss(corase_out, gt) + 2 * dice_loss(re_out, gt) + dice_loss(1 - edge_out, 1 - gt_Sobel) + 0.5*(dice_loss(
        #     corase_out_ori, gt_ori) + dice_loss(corase_out_ori_full, gt_ori_full))
        # else:
        #     flag_white = False
        #     mask_loss = dice_loss(1 - corase_out, 1 - gt) + 2 * dice_loss(1 - re_out, 1 - gt) + dice_loss(edge_out,
        #                                                                                   gt_Sobel) + 0.5 * (
        #                             dice_loss(
        #                                 1 - corase_out_ori, 1 - gt_ori) + dice_loss(1 - corase_out_ori_full, 1 - gt_ori_full))
        # x_fake = torch.concat(image,sobel,thread,output,dim=1)#可能没有batchsize要改dim
        # x_real = torch.concat(image,sobel,thread,target,dim=1)
        model = model.eval()
        self.discriminator_c.train()
        self.discriminator_c.zero_grad()
        # 判别器的loss，判别真实mask时
        #print("x_fake,x_real is",x_fake.shape,x_real.shape)都是1，6，512，512
        D_real_c = self.discriminator_c(x_real)
        D_real_c = D_real_c.mean().sum() * -1#趋于1
        #判别器的loss，判别生成的mask时
        D_fake_c = self.discriminator_c(x_fake)
        D_fake_c = D_fake_c.mean().sum() * 1#趋于0

        D_loss_c = torch.mean(F.relu(1. + D_real_c)) + torch.mean(F.relu(1. + D_fake_c))
        #D_loss_c = D_real_c + D_fake_c
        #print("D_loss_c is",D_loss_c)
        # 判别器网络更新
        self.D_optimizer_c.zero_grad()
        D_loss_c.backward(retain_graph=True)
        self.D_optimizer_c.step()
        self.discriminator_c.eval()
        model = model.train()
        D_fake_c_ = self.discriminator_c(x_fake)
        D_all_c_ = -(torch.mean(D_fake_c_))
        #print("结束")


        #     D_loss_c_all, D_loss_c_full, D_loss_c, D_real_c_full, D_fake_c_full_, D_real_c, D_fake_c_, D_real_r, D_fake_r_ = torch.Tensor([0]).cuda(),torch.Tensor([0]).cuda(),torch.Tensor([0]).cuda(),torch.Tensor([0]).cuda(),torch.Tensor([0]).cuda(),torch.Tensor([0]).cuda(),torch.Tensor([0]).cuda(),torch.Tensor([0]).cuda(),torch.Tensor([0]).cuda()

        return  D_loss_c,D_all_c_ #l1_loss, cross_entropy_loss, mask_loss,