// Copyright (c) Facebook, Inc. and its affiliates.
// 
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once
#include <torch/extension.h>

/*
 * @brief 点云特征采样(Gather Points)
 * 
 * 根据给定的索引idx, 从输入点云特征points中采样, 返回采样后的特征。
 * 常用于根据采样点索引提取对应的特征。
 * 
 * @param points (Tensor) 输入点云特征, 形状为[B, C, N]
 * @param idx    (Tensor) 采样点的索引, 形状为[B, npoint]
 * @return       (Tensor) 采样后的特征, 形状为[B, C, npoint]
 */
at::Tensor gather_points(at::Tensor points, at::Tensor idx);

/*
 * @brief 点云特征采样的反向传播(Gather Points Grad)
 * 
 * 计算gather_points操作的梯度, 将上游梯度grad_out根据采样索引idx累加回原始输入特征points的位置。
 * 用于训练时反向传播, 确保梯度正确传递到原始点云特征。
 * 
 * @param grad_out (Tensor) 上游梯度, 形状为[B, C, npoint]
 * @param idx      (Tensor) 采样点的索引, 形状为[B, npoint]
 * @param n        (int)    原始点的数量N
 * @return         (Tensor) 输入特征points的梯度, 形状为[B, C, N]
 */
at::Tensor gather_points_grad(at::Tensor grad_out, at::Tensor idx, const int n);

/*
 * @brief 最远点采样(Furthest Point Sampling, FPS)
 * 
 * 在输入点云points中, 按照最远点策略采样出nsamples个点的索引。
 * 常用于点云下采样, 保证采样点分布均匀, 覆盖整个点云空间。
 * 
 * @param points   (Tensor) 输入点云坐标, 形状为[B, N, 3]
 * @param nsamples (int)    需要采样的点数
 * @return         (Tensor) 采样点的索引, 形状为[B, nsamples]
 */
at::Tensor furthest_point_sampling(at::Tensor points, const int nsamples);
