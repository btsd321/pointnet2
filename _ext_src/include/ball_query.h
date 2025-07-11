// Copyright (c) Facebook, Inc. and its affiliates.
// 
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once
#include <torch/extension.h>

/*
 * @brief 球查询(Ball Query)操作, 用于点云处理中的邻域搜索。
 * 
 * 该函数用于在点云xyz中, 以new_xyz为中心, 查找半径radius内的邻域点, 最多返回nsample个点的索引。
 * 常用于PointNet++等点云网络的特征提取阶段。
 * 
 * @param new_xyz   (Tensor) 查询中心点的坐标, 形状为[B, npoint, 3]
 * @param xyz       (Tensor) 原始点云的坐标, 形状为[B, N, 3]
 * @param radius    (float)  球查询的半径
 * @param nsample   (int)    每个球内最多采样的点数
 * @return          (Tensor) 返回每个中心点对应的邻域点索引, 形状为[B, npoint, nsample]
 */
at::Tensor ball_query(at::Tensor new_xyz, at::Tensor xyz, const float radius,
                      const int nsample);
