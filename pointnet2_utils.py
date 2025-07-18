import os
import sys
import warnings
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(FILE_DIR)

import torch
from torch.autograd import Function
import torch.nn as nn
import numpy as np
import pytorch_utils as pt_utils

# 以下为可视化点云的工具函数(已注释, 仅供参考)
# def show_points(point_array, color_array=None, radius=3):
#     """
#     可视化点云及其颜色
#     参数:
#         point_array: 点云列表, 每个元素为(N, 3)的numpy数组
#         color_array: 颜色列表, 每个元素为(N, 3)的numpy数组
#         radius: 球体半径
#     """
#     assert isinstance(point_array, list)
#     all_color = None
#     if color_array is not None:
#         assert len(point_array) == len(color_array)
#         all_color = [np.zeros([pnts.shape[0], 3]) for pnts in point_array]
#         for i, c in enumerate(color_array):
#             all_color[i][:] = c
#         all_color = np.concatenate(all_color, axis=0)
#     all_points = np.concatenate(point_array, axis=0)
#     show3d_balls.showpoints(all_points, c_gt=all_color, ballradius=radius)

try:
    import pointnet2_ops._ext as _ext
except ImportError:
    from torch.utils.cpp_extension import load
    import glob
    import os.path as osp
    import os

    warnings.warn("Unable to load pointnet2_ops cpp extension. JIT Compiling.")

    _ext_src_root = osp.join(osp.dirname(__file__), "_ext-src")
    _ext_sources = glob.glob(osp.join(_ext_src_root, "src", "*.cpp")) + glob.glob(
        osp.join(_ext_src_root, "src", "*.cu")
    )
    _ext_headers = glob.glob(osp.join(_ext_src_root, "include", "*"))

    os.environ["TORCH_CUDA_ARCH_LIST"] = "3.7+PTX;5.0;6.0;6.1;6.2;7.0;7.5"
    _ext = load(
        "_ext",
        sources=_ext_sources,
        extra_include_paths=[osp.join(_ext_src_root, "include")],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-Xfatbin", "-compress-all"],
        with_cuda=True,
    )

if False:
    # 类型提示兼容处理
    from typing import *

class RandomDropout(nn.Module):
    """
    随机丢弃特征的模块, 用于防止过拟合。
    """
    def __init__(self, p=0.5, inplace=False):
        super(RandomDropout, self).__init__()
        self.p = p
        self.inplace = inplace

    def forward(self, X):
        """
        前向传播, 随机丢弃部分特征。
        参数:
            X: 输入特征张量
        返回:
            丢弃部分特征后的张量
        """
        theta = torch.Tensor(1).uniform_(0, self.p)[0]
        return pt_utils.feature_dropout_no_scaling(X, theta, self.train, self.inplace)

class FurthestPointSampling(Function):
    """
    最远点采样操作, 返回采样点的索引。
    """
    @staticmethod
    def forward(ctx, xyz, npoint):
        """
        前向传播, 执行最远点采样。
        参数:
            xyz: (B, N, 3) 输入点云坐标
            npoint: int, 采样点数
        返回:
            (B, npoint) 采样点的索引
        """
        return _ext.furthest_point_sampling(xyz, npoint)

    @staticmethod
    def backward(xyz, a=None):
        # 采样操作不可微, 反向传播返回None
        return None, None

furthest_point_sample = FurthestPointSampling.apply

class GatherOperation(Function):
    """
    按照索引采样特征的操作, 支持反向传播。
    """
    @staticmethod
    def forward(ctx, features, idx):
        """
        前向传播, 采样特征。
        参数:
            features: (B, C, N) 输入特征
            idx: (B, npoint) 采样索引
        返回:
            (B, C, npoint) 采样后的特征
        """
        _, C, N = features.size()
        ctx.for_backwards = (idx, C, N)
        return _ext.gather_points(features, idx)

    @staticmethod
    def backward(ctx, grad_out):
        """
        反向传播, 将梯度累加回原始特征。
        参数:
            grad_out: (B, C, npoint) 上游梯度
        返回:
            grad_features: (B, C, N) 原始特征的梯度
            None
        """
        idx, C, N = ctx.for_backwards
        grad_features = _ext.gather_points_grad(grad_out.contiguous(), idx, N)
        return grad_features, None

gather_operation = GatherOperation.apply

class ThreeNN(Function):
    """
    查找每个点的三个最近邻点及其距离。
    """
    @staticmethod
    def forward(ctx, unknown, known):
        """
        前向传播, 查找unknown中每个点在known中的三个最近邻点。
        参数:
            unknown: (B, n, 3) 需要查找的点
            known: (B, m, 3) 已知点
        返回:
            dist: (B, n, 3) 最近三个点的欧氏距离
            idx: (B, n, 3) 最近三个点的索引
        """
        dist2, idx = _ext.three_nn(unknown, known)
        return torch.sqrt(dist2), idx

    @staticmethod
    def backward(ctx, a=None, b=None):
        # 最近邻查找不可微, 反向传播返回None
        return None, None

three_nn = ThreeNN.apply

class ThreeInterpolate(Function):
    """
    三线性插值操作, 用于特征插值。
    """
    @staticmethod
    def forward(ctx, features, idx, weight):
        """
        前向传播, 对输入特征做三线性插值。
        参数:
            features: (B, c, m) 已知点的特征
            idx: (B, n, 3) 三个最近邻的索引
            weight: (B, n, 3) 三个最近邻的插值权重
        返回:
            (B, c, n) 插值后的特征
        """
        B, c, m = features.size()
        n = idx.size(1)
        ctx.three_interpolate_for_backward = (idx, weight, m)
        return _ext.three_interpolate(features, idx, weight)

    @staticmethod
    def backward(ctx, grad_out):
        """
        反向传播, 将梯度累加回原始特征。
        参数:
            grad_out: (B, c, n) 上游梯度
        返回:
            grad_features: (B, c, m) 原始特征的梯度
            None
            None
        """
        idx, weight, m = ctx.three_interpolate_for_backward
        grad_features = _ext.three_interpolate_grad(
            grad_out.contiguous(), idx, weight, m
        )
        return grad_features, None, None

three_interpolate = ThreeInterpolate.apply

class GroupingOperation(Function):
    """
    按照分组索引将特征分组的操作, 支持反向传播。
    """
    @staticmethod
    def forward(ctx, features, idx):
        """
        前向传播, 分组特征。
        参数:
            features: (B, C, N) 输入特征
            idx: (B, npoint, nsample) 分组索引
        返回:
            (B, C, npoint, nsample) 分组后的特征
        """
        B, nfeatures, nsample = idx.size()
        _, C, N = features.size()
        ctx.for_backwards = (idx, N)
        return _ext.group_points(features, idx)

    @staticmethod
    def backward(ctx, grad_out):
        """
        反向传播, 将梯度累加回原始特征。
        参数:
            grad_out: (B, C, npoint, nsample) 上游梯度
        返回:
            grad_features: (B, C, N) 原始特征的梯度
            None
        """
        idx, N = ctx.for_backwards
        grad_features = _ext.group_points_grad(grad_out.contiguous(), idx, N)
        return grad_features, None

grouping_operation = GroupingOperation.apply

class BallQuery(Function):
    """
    球查询操作, 查找每个采样点的邻域点索引。
    """
    @staticmethod
    def forward(ctx, radius, nsample, xyz, new_xyz):
        """
        前向传播, 查找new_xyz中每个点在xyz中的邻域点索引。
        参数:
            radius: float, 球半径
            nsample: int, 最大邻域点数
            xyz: (B, N, 3) 原始点坐标
            new_xyz: (B, npoint, 3) 查询中心点坐标
        返回:
            (B, npoint, nsample) 邻域点索引
        """
        return _ext.ball_query(new_xyz, xyz, radius, nsample)

    @staticmethod
    def backward(ctx, a=None):
        # 球查询不可微, 反向传播返回None
        return None, None, None, None

ball_query = BallQuery.apply

class QueryAndGroup(nn.Module):
    """
    球查询分组模块, 将每个采样点的邻域点特征分组输出。
    """
    def __init__(self, radius, nsample, use_xyz=True):
        """
        初始化球查询分组模块。
        参数:
            radius: 球查询半径
            nsample: 每个球内采样的最大点数
            use_xyz: 是否将xyz坐标拼接到特征中
        """
        super(QueryAndGroup, self).__init__()
        self.radius, self.nsample, self.use_xyz = radius, nsample, use_xyz

    def forward(self, xyz, new_xyz, features=None):
        """
        前向传播, 分组采样点邻域特征。
        参数:
            xyz: (B, N, 3) 原始点坐标
            new_xyz: (B, npoint, 3) 采样点坐标
            features: (B, C, N) 原始点特征(可选)
        返回:
            new_features: (B, 3+C, npoint, nsample) 分组后的特征
        详细说明：
            1. 对每个采样点做球查询, 获得邻域点索引。
            2. 计算邻域点相对采样点的坐标差。
            3. 若有特征, 则拼接坐标差和特征, 否则仅返回坐标差。
        """
        idx = ball_query(self.radius, self.nsample, xyz, new_xyz)
        xyz_trans = xyz.transpose(1, 2).contiguous()
        grouped_xyz = grouping_operation(xyz_trans, idx)  # (B, 3, npoint, nsample)
        grouped_xyz -= new_xyz.transpose(1, 2).unsqueeze(-1)

        if features is not None:
            grouped_features = grouping_operation(features, idx)
            if self.use_xyz:
                new_features = torch.cat(
                    [grouped_xyz, grouped_features], dim=1
                )  # (B, C + 3, npoint, nsample)
            else:
                new_features = grouped_features
        else:
            assert (
                self.use_xyz
            ), "不能既没有特征又不使用xyz作为特征！"
            new_features = grouped_xyz

        return new_features

class GroupAll(nn.Module):
    """
    全局分组模块, 将所有点分为一组。
    """
    def __init__(self, use_xyz=True):
        """
        初始化全局分组模块。
        参数:
            use_xyz: 是否将xyz坐标拼接到特征中
        """
        super(GroupAll, self).__init__()
        self.use_xyz = use_xyz

    def forward(self, xyz, new_xyz, features=None):
        """
        前向传播, 将所有点分为一组。
        参数:
            xyz: (B, N, 3) 原始点坐标
            new_xyz: (B, 1, 3) 虽然传入但实际未用
            features: (B, C, N) 原始点特征(可选)
        返回:
            new_features: (B, C+3, 1, N) 全局分组特征
        详细说明：
            1. 所有点作为一个分组, 输出为(1, N)。
            2. 若有特征则拼接xyz, 否则仅输出xyz。
        """
        grouped_xyz = xyz.transpose(1, 2).unsqueeze(2)
        if features is not None:
            grouped_features = features.unsqueeze(2)
            if self.use_xyz:
                new_features = torch.cat(
                    [grouped_xyz, grouped_features], dim=1
                )  # (B, 3 + C, 1, N)
            else:
                new_features = grouped_features
        else:
            new_features = grouped_xyz

        return new_features
