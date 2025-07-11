import os
import sys
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(FILE_DIR)
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_utils as pt_utils

import pointnet2_utils

if False:
    # 为了类型提示而不引入typing依赖
    from typing import *


class _PointnetSAModuleBase(nn.Module):
    def __init__(self):
        super(_PointnetSAModuleBase, self).__init__()
        self.npoint = None
        self.groupers = None
        self.mlps = None

    def forward(self, xyz, features=None):
        """
        点集抽象模块的前向传播

        参数:
            xyz : torch.Tensor
                (B, N, 3) 输入点的空间坐标
            features : torch.Tensor
                (B, N, C) 输入点的特征(可选)

        返回:
            new_xyz : torch.Tensor
                (B, npoint, 3) 采样后的点的空间坐标
            new_features : torch.Tensor
                (B, 所有尺度输出通道之和, npoint) 采样后的点的特征
        详细说明：
            1. 先对输入点云进行最远点采样, 得到采样点new_xyz。
            2. 对每个尺度分别进行分组和特征聚合(球查询+MLP+池化)。
            3. 多尺度特征拼接输出。
        """
        new_features_list = []

        xyz_flipped = xyz.transpose(1, 2).contiguous()
        new_xyz = (
            pointnet2_utils.gather_operation(
                xyz_flipped, pointnet2_utils.furthest_point_sample(xyz, self.npoint)
            )
            .transpose(1, 2)
            .contiguous()
            if self.npoint is not None
            else None
        )

        for i in range(len(self.groupers)):
            # 分组操作, 得到每个采样点邻域的特征
            new_features = self.groupers[i](
                xyz, new_xyz, features
            )  # (B, C, npoint, nsample)

            # 通过MLP提取局部特征
            new_features = self.mlps[i](new_features)  # (B, mlp[-1], npoint, nsample)
            # 对邻域特征做最大池化, 得到每个采样点的局部描述
            new_features = F.max_pool2d(
                new_features, kernel_size=[1, new_features.size(3)]
            )  # (B, mlp[-1], npoint, 1)
            new_features = new_features.squeeze(-1)  # (B, mlp[-1], npoint)

            new_features_list.append(new_features)

        # 多尺度特征拼接
        return new_xyz, torch.cat(new_features_list, dim=1)


class PointnetSAModuleMSG(_PointnetSAModuleBase):
    r"""
    PointNet++多尺度分组(MSG)点集抽象层

    参数说明
    ----------
    npoint : int
        采样点数
    radii : list of float32
        每个尺度的球查询半径
    nsamples : list of int32
        每个尺度的邻域采样点数
    mlps : list of list of int32
        每个尺度的MLP结构
    bn : bool
        是否使用BatchNorm
    use_xyz : bool
        是否将xyz坐标拼接到特征中
    """

    def __init__(self, npoint, radii, nsamples, mlps, bn=True, use_xyz=True):
        super(PointnetSAModuleMSG, self).__init__()

        assert len(radii) == len(nsamples) == len(mlps)

        self.npoint = npoint
        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()
        for i in range(len(radii)):
            radius = radii[i]
            nsample = nsamples[i]
            # 构建每个尺度的分组器(球查询)
            self.groupers.append(
                pointnet2_utils.QueryAndGroup(radius, nsample, use_xyz=use_xyz)
                if npoint is not None
                else pointnet2_utils.GroupAll(use_xyz)
            )
            mlp_spec = mlps[i]
            if use_xyz:
                mlp_spec[0] += 3  # 输入特征拼接xyz

            # 构建每个尺度的MLP
            self.mlps.append(pt_utils.SharedMLP(mlp_spec, bn=bn))


class PointnetSAModule(PointnetSAModuleMSG):
    r"""
    PointNet++单尺度分组(SSG)点集抽象层

    参数说明
    ----------
    npoint : int
        采样点数
    radius : float
        球查询半径
    nsample : int
        邻域采样点数
    mlp : list
        MLP结构
    bn : bool
        是否使用BatchNorm
    use_xyz : bool
        是否将xyz坐标拼接到特征中
    """

    def __init__(
        self, mlp, npoint=None, radius=None, nsample=None, bn=True, use_xyz=True
    ):
        super(PointnetSAModule, self).__init__(
            mlps=[mlp],
            npoint=npoint,
            radii=[radius],
            nsamples=[nsample],
            bn=bn,
            use_xyz=use_xyz,
        )


class PointnetFPModule(nn.Module):
    r"""
    特征传播(Feature Propagation)模块, 用于特征插值和融合

    参数说明
    ----------
    mlp : list
        MLP结构
    bn : bool
        是否使用BatchNorm
    """

    def __init__(self, mlp, bn=True):
        super(PointnetFPModule, self).__init__()
        self.mlp = pt_utils.SharedMLP(mlp, bn=bn)

    def forward(self, unknown, known, unknow_feats, known_feats):
        """
        特征传播模块前向传播

        参数:
            unknown : torch.Tensor
                (B, n, 3) 目标点的空间坐标(需要插值的点)
            known : torch.Tensor
                (B, m, 3) 已知点的空间坐标(有特征的点)
            unknow_feats : torch.Tensor
                (B, C1, n) 目标点的特征(可选, 来自上一级FP)
            known_feats : torch.Tensor
                (B, C2, m) 已知点的特征

        返回:
            new_features : torch.Tensor
                (B, mlp[-1], n) 插值融合后的特征
        详细说明：
            1. 若known不为None, 则对known_feats做三线性插值, 获得unknown点的特征。
            2. 若unknow_feats不为None, 则与插值特征拼接。
            3. 拼接后通过MLP提升特征表达能力。
        """
        if known is not None:
            # 三近邻查找与插值
            dist, idx = pointnet2_utils.three_nn(unknown, known)
            dist_recip = 1.0 / (dist + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm

            interpolated_feats = pointnet2_utils.three_interpolate(
                known_feats, idx, weight
            )
        else:
            # 若无已知点, 则直接扩展特征
            interpolated_feats = known_feats.expand(
                *(known_feats.size()[0:2] + [unknown.size(1)])
            )

        if unknow_feats is not None:
            # 拼接上一级FP特征
            new_features = torch.cat(
                [interpolated_feats, unknow_feats], dim=1
            )  # (B, C2 + C1, n)
        else:
            new_features = interpolated_feats

        new_features = new_features.unsqueeze(-1)
        new_features = self.mlp(new_features)

        return new_features.squeeze(-1)


if __name__ == "__main__":
    from torch.autograd import Variable

    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    xyz = Variable(torch.randn(2, 9, 3).cuda(), requires_grad=True)
    xyz_feats = Variable(torch.randn(2, 9, 6).cuda(), requires_grad=True)

    # 测试MSG模块
    test_module = PointnetSAModuleMSG(
        npoint=2, radii=[5.0, 10.0], nsamples=[6, 3], mlps=[[9, 3], [9, 6]]
    )
    test_module.cuda()
    print(test_module(xyz, xyz_feats))

    # # 测试FP模块
    # test_module = PointnetFPModule(mlp=[6, 6])
    # test_module.cuda()
    # from torch.autograd import gradcheck
    # inputs = (xyz, xyz, None, xyz_feats)
    # test = gradcheck(test_module, inputs, eps=1e-6, atol=1e-4)
    # print(test)

    for _ in range(1):
        _, new_features = test_module(xyz, xyz_feats)
        new_features.backward(torch.cuda.FloatTensor(*new_features.size()).fill_(1))
        print(new_features)
        print(xyz.grad)
