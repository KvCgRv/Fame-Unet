import torch
from torch import nn
from torch.nn import functional
from models.Discriminator import Discriminator
from utils.eval_utils import multiclass_dice_coeff, dice_coeff, build_target
from src.cldice import soft_cldice
import numpy as np
import cv2
def dice_loss(x: torch.Tensor, target: torch.Tensor, multiclass: bool = False, ignore_index: int = -100):
    # Dice loss (objective to minimize) between 0 and 1
    x = functional.softmax(x, dim=1)
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(x, target, ignore_index=ignore_index)


# def criterion(inputs, target, loss_weight=None, num_classes: int = 2, dice: bool = True, ignore_index: int = -100):
#     losses = {}
#     for name, x in inputs.items():
#         # 忽略target中值为255的像素，255的像素是目标边缘或者padding填充
#         # loss = functional.cross_entropy(x, target, ignore_index=ignore_index, weight=loss_weight)
#         target = torch.where(target == 255, 0, target)
#         loss = functional.cross_entropy(x, target, weight=loss_weight)
#         if dice is True:
#             dice_target = build_target(target, num_classes, ignore_index)
#             loss += dice_loss(x, dice_target, multiclass=True, ignore_index=ignore_index)
#         losses[name] = loss
#     losses["levelset"] = level_set_loss_compute(inputs)
#     # losses["levelset"] = level_set_loss_compute(inputs, target)
#     if len(losses) == 1:
#         return losses['out']
#
#     return losses["out"] + 1e-6 * losses["levelset"]
#     # return losses['out'] + 0.5 * losses['aux']


def phase_loss_compute(model,pred,target):
    phase_loss = model(pred,target)
    return phase_loss

def criterion(inputs, target, Phase_model, loss_weight=None, num_classes: int = 2, dice: bool = True, ignore_index: int = -100):
    losses = {}
    x = inputs["out"]
    target = torch.where(target == 255, 0, target)#1,255变成1,0,此时血管1白色，背景0黑色
    # losses["ce_loss"] = functional.cross_entropy(x, target, weight=loss_weight)
    losses["ce_loss"] =    phase_loss_compute(Phase_model,x,target) + functional.cross_entropy(x, target, weight=loss_weight)
    losses["level_set_loss"] = level_set_loss_compute_supervised(inputs, target)
    #losses["level_set_loss"] =  phase_loss_compute(Phase_model,x,target)
    #losses["level_set_loss"] = torch.zeros(1).cuda()
    if dice is True:
        dice_target = build_target(target, num_classes, ignore_index)
        losses["dice_loss"] = dice_loss(x, dice_target, multiclass=True, ignore_index=ignore_index) 
        #losses["dice_loss"] = torch.zeros(1).cuda()
    return losses


def criterion_supervised(inputs, target, loss_weight=None, num_classes: int = 2, dice: bool = True,
                         ignore_index: int = -100):
    losses = {}
    x = inputs["out"]
    target = torch.where(target == 255, 0, target)
    losses["ce_loss"] = functional.cross_entropy(x, target, weight=loss_weight)
    losses["level_set_loss"] = torch.zeros(1).cuda()
    if dice is True:
        dice_target = build_target(target, num_classes, ignore_index)
        losses["dice_loss"] = dice_loss(x, dice_target, multiclass=True, ignore_index=ignore_index)
    return losses

import numpy as np


def get_sobel(tensor):
    # 确保输入tensor是CPU上的并且是float类型
    tensor = tensor.squeeze().cpu().float()
    #print("shape of tensor is", tensor.shape)
    # 转换tensor到numpy数组
    gt_np = tensor.numpy().astype(np.uint8)

    # 将RGB图像转换为灰度图像
    gray = cv2.cvtColor(gt_np, cv2.COLOR_RGB2GRAY)

    # 使用OpenCV计算Sobel算子
    x = cv2.Sobel(gray, cv2.CV_16S, 1, 0)
    y = cv2.Sobel(gray, cv2.CV_16S, 0, 1)

    # 取绝对值并合并梯度
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

    # 将结果转换回tensor
    dst_tensor = torch.from_numpy(dst.astype(np.float32))

    return dst_tensor


def get_otsu(tensor):
    # 确保输入tensor是CPU上的并且是float类型
    tensor = tensor.squeeze().cpu().float()
    #print("shape of tensor is", tensor.shape)
    # 转换tensor到numpy数组
    img_np = tensor.numpy().astype(np.uint8)

    # 检查图像是否为RGB图像（3通道）
    if img_np.ndim == 3 and img_np.shape[2] == 3:
        # 将RGB图像转换为灰度图像
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    else:
        # 如果不是RGB图像，直接使用输入图像
        gray = img_np

    # 应用Otsu二值化方法
    ret, th2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 将二值化后的numpy数组转换回tensor
    otsu_tensor = torch.from_numpy(th2.astype(np.uint8)).float()  # 转换为float类型以保持一致性

    return otsu_tensor


def criterion_gan(model,image,outputs, target, Phase_model, loss_doc,loss_weight=None, num_classes: int = 2, dice: bool = True,ignore_index: int = -100):
    losses = {}
    x = outputs["out"].cuda()
    target = torch.where(target == 255, 1, target)
    x_clone = x.clone().softmax(1).argmax(1).cuda()  # 获得网络的输出
    image = image.squeeze().permute(1, 2, 0)  # B,C,H,W->B,H,W,C
    #image = image.squeeze().permute(0, 2, 3, 1)  # B,C,H,W->B,H,W,C
    x_sobel = get_sobel(image)
    x_sobel = x_sobel.unsqueeze(0).cuda()
    x_thread = get_otsu(image)
    x_thread = x_thread.unsqueeze(0).cuda()
    image = image.permute(2, 0, 1)# B,H,W,C->B,C,H,W
    #image = image.permute(0, 3, 1, 2)# B,H,W,C->B,C,H,W
    x_fake = torch.concat((image, x_sobel, x_thread, x_clone), dim=0)  # 可能没有batchsize要改dim
    x_real = torch.concat((image, x_sobel, x_thread, target), dim=0)
    # x_fake = torch.concat((image, x_clone), dim=0)  # 去掉阈值对应的discriminator要从6改成5
    # x_real = torch.concat((image, target), dim=0)
    x_fake = x_fake.unsqueeze(0)
    x_real = x_real.unsqueeze(0)
    # with torch.no_grad():
    D_loss_c, D_fake_c = loss_doc(model,x_fake, x_real)
    x_2 = x.float()
    target = target.float()
    target2 = target.long()
    # losses["gan_loss"] = D_fake_c
    # losses["ce_loss"] =   phase_loss_compute(Phase_model,x,target) + functional.cross_entropy(x, target, weight=loss_weight) + functional.l1_loss(x_clone,target)
    losses["ce_loss"] = functional.cross_entropy(x_2, target2, weight=loss_weight) + phase_loss_compute(Phase_model,x,target2) +  0.0005 * D_fake_c
    losses["level_set_loss"] = level_set_loss_compute_supervised(outputs, target)
    #print("结束2")
    if dice is True:
        cl_dice_loss = soft_cldice(smooth = 1.)
        # x_3 = x_2.unsqueeze(0)
        target = target.long()
        dice_target = build_target(target, num_classes, ignore_index)
        losses["dice_loss"] = dice_loss(x_2, dice_target, multiclass=True, ignore_index=ignore_index) + cl_dice_loss(x,target)#是否带cldice？
    else:
        losses["dice_loss"] = torch.Tensor([0]).cuda()
    return losses



# class LevelSetLoss(nn.Module):
#     def __init__(self):
#         super(LevelSetLoss, self).__init__()
#
#     def forward(self, measurement, softmax_output):
#         # input size = batch x channel x height x width
#         outshape = softmax_output.shape
#         tarshape = measurement.shape
#         loss = 0.0
#         for ich in range(tarshape[1]):
#             target_ = torch.unsqueeze(measurement[:, ich], 1)
#             target_ = target_.expand(tarshape[0], outshape[1], tarshape[2], tarshape[3])
#             pcentroid = torch.sum(target_ * softmax_output, (2, 3)) / torch.sum(softmax_output, (2, 3))
#             pcentroid = pcentroid.view(tarshape[0], outshape[1], 1, 1)
#             plevel = target_ - pcentroid.expand(tarshape[0], outshape[1], tarshape[2], tarshape[3])
#             pLoss = plevel * plevel * softmax_output
#             loss += torch.sum(pLoss)
#         return loss


class GradientLoss2d(nn.Module):
    def __init__(self, penalty='l1'):
        super(GradientLoss2d, self).__init__()
        self.penalty = penalty

    def forward(self, input):
        dH = torch.abs(input[:, :, 1:, :] - input[:, :, :-1, :])
        dW = torch.abs(input[:, :, :, 1:] - input[:, :, :, :-1])
        # l2损失
        if (self.penalty == "l2"):
            dH = dH ** 2
            dW = dW ** 2

        loss = torch.sum(dH) + torch.sum(dW)
        return loss


# def level_set_loss_compute(net_output: dict, target: torch.Tensor):
def level_set_loss_compute(net_output: dict):
    """
    返回水平集损失
    :param net_output:  batch * class * height * weight
    :return: tensor, loss value
    """
    # # 对target进行处理
    # target = torch.unsqueeze(target, 1)
    # target_back = 1 - target
    # target = torch.cat([target, target_back], dim=1)

    softmaxed = net_output["out"].softmax(1)
    argmaxed = net_output["out"].argmax(1)
    # 将channel0维也就是背景全部设置为负
    back_ground = -softmaxed[:, 0, :, :]
    fore = softmaxed[:, 1:, :, :]
    fore_ground = torch.sum(fore, dim=(1,), keepdim=True)
    fore_ground = fore_ground.squeeze(1)

    # 两者得到，开始计算水平集损失
    measurement = torch.where(argmaxed == 0, back_ground, fore_ground)
    softmax_output = net_output["out"].softmax(1)

    loss = 0.0

    measurement_multi_channel = torch.unsqueeze(measurement, 1)
    measurement_multi_channel = measurement_multi_channel.expand(softmax_output.shape[0], softmax_output.shape[1],
                                                                 softmax_output.shape[2], softmax_output.shape[3])
    # 这里是自监督，预测的前景和背景靠近自身
    pcentroid = torch.sum(measurement_multi_channel * softmax_output, (2, 3)) / torch.sum(softmax_output, (2, 3))

    # # 这里加了groundtruth，在groundtruth的限定下，前景逼近前景，背景逼近背景
    # pcentroid = torch.sum(measurement_multi_channel * target, (2, 3)) / torch.sum(target, (2, 3))

    pcentroid = pcentroid.view(softmax_output.shape[0], softmax_output.shape[1], 1, 1)
    plevel = measurement_multi_channel - pcentroid.expand(softmax_output.shape[0], softmax_output.shape[1],
                                                          softmax_output.shape[2], softmax_output.shape[3])
    pLoss = plevel * plevel * softmax_output
    loss = loss + torch.sum(pLoss)
    # loss.backward()
    return loss


def level_set_loss_compute_supervised(net_output: dict, target: torch.Tensor):
    """
    返回水平集损失（有监督）
    :param net_output:  batch * class * height * weight
    :return: tensor, loss value
    :param target: ground truth
    :return:
    """

    # # 对target进行处理
    # target = torch.unsqueeze(target, 1)
    # target_back = 1 - target
    # target = torch.cat([target, target_back], dim=1)

    softmaxed = net_output["out"].softmax(1)
    argmaxed = net_output["out"].argmax(1)
    # 将channel0维也就是背景全部设置为负
    back_ground = -softmaxed[:, 0, :, :]
    fore = softmaxed[:, 1:, :, :]
    fore_ground = torch.sum(fore, dim=(1,), keepdim=True)
    fore_ground = fore_ground.squeeze(1)

    # 两者得到，开始计算水平集损失
    measurement = torch.where(argmaxed == 0, back_ground, fore_ground)
    softmax_output = net_output["out"].softmax(1)

    loss = 0.0

    measurement_multi_channel = torch.unsqueeze(measurement, 1)
    measurement_multi_channel = measurement_multi_channel.expand(softmax_output.shape[0], softmax_output.shape[1],
                                                                 softmax_output.shape[2], softmax_output.shape[3])
    # 这里是自监督，预测的前景和背景靠近自身
    # pcentroid = torch.sum(measurement_multi_channel * softmax_output, (2, 3)) / torch.sum(softmax_output, (2, 3))

    # 这里加了groundtruth，在groundtruth的限定下，前景逼近前景，背景逼近背景
    target = target.unsqueeze(dim=1)
    target = torch.cat([1 - target, target], dim=1)
    pcentroid = torch.sum(measurement_multi_channel * target, (2, 3)) / torch.sum(target, (2, 3))

    pcentroid = pcentroid.view(softmax_output.shape[0], softmax_output.shape[1], 1, 1)
    plevel = measurement_multi_channel - pcentroid.expand(softmax_output.shape[0], softmax_output.shape[1],
                                                          softmax_output.shape[2], softmax_output.shape[3])
    pLoss = plevel * plevel * target
    loss = loss + torch.sum(pLoss)
    # loss.backward()
    return loss


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # # 仿制输入
    # dummy_net_output = torch.autograd.Variable(torch.sigmoid(torch.randn(2, 2, 8, 16)), requires_grad=True).to(device)
    # # 仿制groundtruth
    # dummy_truth = torch.autograd.Variable(torch.ones_like(dummy_net_output)).to(device)
    # print('Input Size :', dummy_net_output.size())

    '''
    dummy_net_output = torch.tensor(
        [[[[-1.1, 1.0], [2.1, -2.0]], [[1.2, -1.2], [2.0, -2.0]], [[1.2, -1.2], [2.0, -2.0]]],
         [[[-1.1, 1.0], [2.1, -2.0]], [[1.2, -1.2], [2.0, -2.0]], [[1.2, -1.2], [2.0, -2.0]]]],
        requires_grad=True).to(device)    
    '''
    dummy_net_output = torch.tensor(
        [[[[-1.1, 1.0], [2.1, -2.0]], [[1.2, -1.2], [2.0, -2.0]], ],
         [[[-1.1, 1.0], [2.1, -2.0]], [[1.2, -1.2], [2.0, -2.0]], ]],
        requires_grad=True).to(device)
    dummy_truth = torch.tensor(
        [[[1, 0], [0, 1]],
         [[1, 1], [0, 0]]], ).to(device)
    # # 评价标准criteria
    # criteria = LevelSetLoss()
    # loss = criteria(dummy_net_output, dummy_truth)
    dummy_input = {}
    dummy_input["out"] = dummy_net_output
    criteria = level_set_loss_compute(dummy_input)
    # criteria = level_set_loss_compute(dummy_input, dummy_truth)

    # print('Loss Value :', loss)
    print('Loss Value :', criteria)
