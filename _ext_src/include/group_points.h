// Copyright (c) Facebook, Inc. and its affiliates.
// 
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once
#include <torch/extension.h>

/*
 * @brief 点云分组操作(Group Points)
 * 
 * 根据给定的索引idx, 从输入点云特征points中采样, 返回分组后的特征。
 * 常用于PointNet++等点云网络的局部特征聚合阶段, 实现对每个采样中心点的邻域特征提取。
 * 
 * @param points (Tensor) 输入点云特征, 形状为[B, C, N], B为batch, C为通道数, N为点数
 * @param idx    (Tensor) 分组采样的索引, 形状为[B, npoint, nsample], npoint为采样中心点数, nsample为每个中心的邻域点数
 * @return       (Tensor) 分组后的特征, 形状为[B, C, npoint, nsample]
 */
at::Tensor group_points(at::Tensor points, at::Tensor idx);

/*
 * @brief 点云分组操作的反向传播(Group Points Grad)
 * 
 * 计算group_points操作的梯度, 将上游梯度grad_out根据采样索引idx累加回原始输入特征points的位置。
 * 用于训练时反向传播, 确保梯度正确传递到原始点云特征。
 * 
 * @param grad_out (Tensor) 上游梯度, 形状为[B, C, npoint, nsample]
 * @param idx      (Tensor) 分组采样的索引, 形状为[B, npoint, nsample]
 * @param n        (int)    原始点的数量N
 * @return         (Tensor) 输入特征points的梯度, 形状为[B, C, N]
 */
at::Tensor group_points_grad(at::Tensor grad_out, at::Tensor idx, const int n);
