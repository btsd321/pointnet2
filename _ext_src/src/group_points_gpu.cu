// Copyright (c) Facebook, Inc. and its affiliates.
// 
// 本源码遵循MIT协议, 详见根目录下的LICENSE文件。

#include <stdio.h>
#include <stdlib.h>

#include "cuda_utils.h"

/*
 * @brief 点云分组操作的CUDA核函数
 * 
 * 根据给定的采样索引idx, 从输入点云特征points中采样, 返回分组后的特征out。
 * 常用于PointNet++等点云网络的局部特征聚合阶段, 实现对每个采样中心点的邻域特征提取。
 * 
 * @param b        批量大小(batch size)
 * @param c        特征通道数
 * @param n        每批次点云的点数
 * @param npoints  每批次采样中心点的数量
 * @param nsample  每个中心点的邻域采样点数
 * @param points   输入点云特征, 形状为(b, c, n)
 * @param idx      分组采样的索引, 形状为(b, npoints, nsample)
 * @param out      输出分组后的特征, 形状为(b, c, npoints, nsample)
 */
__global__ void group_points_kernel(int b, int c, int n, int npoints,
                                    int nsample,
                                    const float *__restrict__ points,
                                    const int *__restrict__ idx,
                                    float *__restrict__ out) {
  // 当前处理的batch索引
  int batch_index = blockIdx.x;
  // 指针偏移到当前batch的数据
  points += batch_index * n * c;
  idx += batch_index * npoints * nsample;
  out += batch_index * npoints * nsample * c;

  // 计算当前线程负责的特征通道和采样中心点的索引
  const int index = threadIdx.y * blockDim.x + threadIdx.x;
  const int stride = blockDim.y * blockDim.x;
  // 遍历所有通道和采样中心点, 每个线程负责间隔stride的任务
  for (int i = index; i < c * npoints; i += stride) {
    const int l = i / npoints;   // 当前特征通道
    const int j = i % npoints;   // 当前采样中心点
    // 遍历每个中心点的邻域采样
    for (int k = 0; k < nsample; ++k) {
      int ii = idx[j * nsample + k]; // 采样点在原始点云中的索引
      // 将采样到的特征写入输出
      out[(l * npoints + j) * nsample + k] = points[l * n + ii];
    }
  }
}

/*
 * @brief 点云分组操作CUDA核函数的包装器
 * 
 * 设置CUDA流和核函数参数, 并调用group_points_kernel执行分组采样操作。
 * 
 * @param b        批量大小
 * @param c        特征通道数
 * @param n        每批次点云的点数
 * @param npoints  采样中心点数量
 * @param nsample  每个中心点的邻域采样点数
 * @param points   输入点云特征
 * @param idx      分组采样的索引
 * @param out      输出分组后的特征
 */
void group_points_kernel_wrapper(int b, int c, int n, int npoints, int nsample,
                                 const float *points, const int *idx,
                                 float *out) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // 启动CUDA核函数, 每个batch一个block, block内线程数根据npoints和c自适应
  group_points_kernel<<<b, opt_block_config(npoints, c), 0, stream>>>(
      b, c, n, npoints, nsample, points, idx, out);

  CUDA_CHECK_ERRORS();
}

/*
 * @brief 点云分组操作反向传播的CUDA核函数
 * 
 * 计算group_points操作的梯度, 将上游梯度grad_out根据采样索引idx累加回原始输入特征grad_points的位置。
 * 用于训练时反向传播, 确保梯度正确传递到原始点云特征。
 * 
 * @param b           批量大小
 * @param c           特征通道数
 * @param n           每批次点云的点数
 * @param npoints     采样中心点数量
 * @param nsample     每个中心点的邻域采样点数
 * @param grad_out    上游梯度, 形状为(b, c, npoints, nsample)
 * @param idx         分组采样的索引, 形状为(b, npoints, nsample)
 * @param grad_points 输出, 输入特征的梯度, 形状为(b, c, n)
 */
__global__ void group_points_grad_kernel(int b, int c, int n, int npoints,
                                         int nsample,
                                         const float *__restrict__ grad_out,
                                         const int *__restrict__ idx,
                                         float *__restrict__ grad_points) {
  // 当前处理的batch索引
  int batch_index = blockIdx.x;
  // 指针偏移到当前batch的数据
  grad_out += batch_index * npoints * nsample * c;
  idx += batch_index * npoints * nsample;
  grad_points += batch_index * n * c;

  // 计算当前线程负责的特征通道和采样中心点的索引
  const int index = threadIdx.y * blockDim.x + threadIdx.x;
  const int stride = blockDim.y * blockDim.x;
  // 遍历所有通道和采样中心点, 每个线程负责间隔stride的任务
  for (int i = index; i < c * npoints; i += stride) {
    const int l = i / npoints;   // 当前特征通道
    const int j = i % npoints;   // 当前采样中心点
    // 遍历每个中心点的邻域采样
    for (int k = 0; k < nsample; ++k) {
      int ii = idx[j * nsample + k]; // 采样点在原始点云中的索引
      // 使用原子加操作将梯度累加到对应的输入特征位置
      atomicAdd(grad_points + l * n + ii,
                grad_out[(l * npoints + j) * nsample + k]);
    }
  }
}

/*
 * @brief 点云分组操作反向传播CUDA核函数的包装器
 * 
 * 设置CUDA流和核函数参数, 并调用group_points_grad_kernel执行梯度累加操作。
 * 
 * @param b           批量大小
 * @param c           特征通道数
 * @param n           每批次点云的点数
 * @param npoints     采样中心点数量
 * @param nsample     每个中心点的邻域采样点数
 * @param grad_out    上游梯度
 * @param idx         分组采样的索引
 * @param grad_points 输出, 输入特征的梯度
 */
void group_points_grad_kernel_wrapper(int b, int c, int n, int npoints,
                                      int nsample, const float *grad_out,
                                      const int *idx, float *grad_points) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // 启动CUDA核函数, 每个batch一个block, block内线程数根据npoints和c自适应
  group_points_grad_kernel<<<b, opt_block_config(npoints, c), 0, stream>>>(
      b, c, n, npoints, nsample, grad_out, idx, grad_points);

  CUDA_CHECK_ERRORS();
}
