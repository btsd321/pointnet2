// Copyright (c) Facebook, Inc. and its affiliates.
// 
// 本源码遵循MIT协议, 详见根目录下的LICENSE文件。

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "cuda_utils.h"

/*
 * @brief 球查询CUDA核函数
 * 
 * 在点云xyz中, 以new_xyz为中心, 查找半径radius内的邻域点, 最多返回nsample个点的索引。
 * 用于点云处理中的邻域搜索, 常见于PointNet++等网络。
 * 
 * @param b        批量大小(batch size)
 * @param n        每批次点云的点数
 * @param m        每批次查询中心点的数量
 * @param radius   球查询半径
 * @param nsample  每个球内最多采样的点数
 * @param new_xyz  查询中心点的坐标, 形状为(b, m, 3)
 * @param xyz      原始点云的坐标, 形状为(b, n, 3)
 * @param idx      输出, 每个中心点对应的邻域点索引, 形状为(b, m, nsample)
 */
__global__ void query_ball_point_kernel(int b, int n, int m, float radius,
                                        int nsample,
                                        const float *__restrict__ new_xyz,
                                        const float *__restrict__ xyz,
                                        int *__restrict__ idx) {
  // 计算当前处理的batch索引
  int batch_index = blockIdx.x;
  // 指针偏移到当前batch的数据
  xyz += batch_index * n * 3;
  new_xyz += batch_index * m * 3;
  idx += m * nsample * batch_index;

  // 当前线程负责的中心点索引起点
  int index = threadIdx.x;
  // 每个线程的步长
  int stride = blockDim.x;

  float radius2 = radius * radius;  // 预先计算半径平方, 便于距离比较
  // 遍历本batch的所有中心点, 每个线程负责间隔stride的中心点
  for (int j = index; j < m; j += stride) {
    float new_x = new_xyz[j * 3 + 0];
    float new_y = new_xyz[j * 3 + 1];
    float new_z = new_xyz[j * 3 + 2];
    // cnt用于统计已找到的邻域点数
    for (int k = 0, cnt = 0; k < n && cnt < nsample; ++k) {
      float x = xyz[k * 3 + 0];
      float y = xyz[k * 3 + 1];
      float z = xyz[k * 3 + 2];
      // 计算欧氏距离的平方
      float d2 = (new_x - x) * (new_x - x) + (new_y - y) * (new_y - y) +
                 (new_z - z) * (new_z - z);
      // 如果距离小于半径, 说明k点在球内
      if (d2 < radius2) {
        // 第一个找到的点, 先将所有位置填为该点索引, 保证至少有一个有效索引
        if (cnt == 0) {
          for (int l = 0; l < nsample; ++l) {
            idx[j * nsample + l] = k;
          }
        }
        // 记录当前找到的点的索引
        idx[j * nsample + cnt] = k;
        ++cnt;
      }
    }
  }
}

/*
 * @brief 球查询CUDA核函数的包装器
 * 
 * 负责设置CUDA流和核函数参数, 并调用核函数执行球查询操作。
 * 
 * @param b        批量大小(batch size)
 * @param n        每批次点云的点数
 * @param m        每批次查询中心点的数量
 * @param radius   球查询半径
 * @param nsample  每个球内最多采样的点数
 * @param new_xyz  查询中心点的坐标, 形状为(b, m, 3)
 * @param xyz      原始点云的坐标, 形状为(b, n, 3)
 * @param idx      输出, 每个中心点对应的邻域点索引, 形状为(b, m, nsample)
 */
void query_ball_point_kernel_wrapper(int b, int n, int m, float radius,
                                     int nsample, const float *new_xyz,
                                     const float *xyz, int *idx) {
  // 获取当前CUDA流
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  // 启动核函数, 每个batch一个block, block内线程数根据m自适应
  query_ball_point_kernel<<<b, opt_n_threads(m), 0, stream>>>(
      b, n, m, radius, nsample, new_xyz, xyz, idx);

  // 检查CUDA错误
  CUDA_CHECK_ERRORS();
}
