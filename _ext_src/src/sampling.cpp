// Copyright (c) Facebook, Inc. and its affiliates.
// 
// 本源码遵循MIT协议, 详见根目录下的LICENSE文件。

#include "sampling.h"
#include "utils.h"

// CUDA核函数包装器声明, 具体实现在.cu文件中
void gather_points_kernel_wrapper(int b, int c, int n, int npoints,
                                  const float *points, const int *idx,
                                  float *out);
void gather_points_grad_kernel_wrapper(int b, int c, int n, int npoints,
                                       const float *grad_out, const int *idx,
                                       float *grad_points);
void furthest_point_sampling_kernel_wrapper(int b, int n, int m,
                                            const float *dataset, float *temp,
                                            int *idxs);

/*
 * @brief 点云特征采样主函数(PyTorch接口)
 * 
 * 根据给定的采样索引idx, 从输入点云特征points中采样, 返回采样后的特征。
 * 常用于根据采样点索引提取对应的特征。
 * 
 * @param points (Tensor) 输入点云特征, 形状为[B, C, N]
 * @param idx    (Tensor) 采样点的索引, 形状为[B, npoints]
 * @return       (Tensor) 采样后的特征, 形状为[B, C, npoints]
 */
at::Tensor gather_points(at::Tensor points, at::Tensor idx) {
  // 检查输入张量是否为连续内存且类型正确
  CHECK_CONTIGUOUS(points);
  CHECK_CONTIGUOUS(idx);
  CHECK_IS_FLOAT(points);
  CHECK_IS_INT(idx);

  // 如果points在CUDA上, 则idx也必须在CUDA上
  if (IS_CUDA_TENSOR(points)) {
    CHECK_CUDA(idx);
  }

  // 创建输出张量, 初始化为0, 形状为[B, C, npoints]
  at::Tensor output =
      torch::zeros({points.size(0), points.size(1), idx.size(1)},
                   at::device(points.device()).dtype(at::ScalarType::Float));

  // 调用CUDA核函数进行特征采样
  if (IS_CUDA_TENSOR(points)) {
    gather_points_kernel_wrapper(points.size(0), points.size(1), points.size(2),
                                 idx.size(1), TENSOR_DATA_PTR(points, float),
                                 TENSOR_DATA_PTR(idx, int), TENSOR_DATA_PTR(output, float));
  } else {
    // 仅支持CUDA实现, CPU暂不支持
    AT_CHECK(false, "CPU not supported");
  }

  // 返回采样后的特征
  return output;
}

/*
 * @brief 点云特征采样反向传播主函数(PyTorch接口)
 * 
 * 计算gather_points操作的梯度, 将上游梯度grad_out根据采样索引idx累加回原始输入特征points的位置。
 * 用于训练时反向传播, 确保梯度正确传递到原始点云特征。
 * 
 * @param grad_out (Tensor) 上游梯度, 形状为[B, C, npoints]
 * @param idx      (Tensor) 采样点的索引, 形状为[B, npoints]
 * @param n        (int)    原始点的数量N
 * @return         (Tensor) 输入特征points的梯度, 形状为[B, C, N]
 */
at::Tensor gather_points_grad(at::Tensor grad_out, at::Tensor idx,
                              const int n) {
  // 检查输入张量是否为连续内存且类型正确
  CHECK_CONTIGUOUS(grad_out);
  CHECK_CONTIGUOUS(idx);
  CHECK_IS_FLOAT(grad_out);
  CHECK_IS_INT(idx);

  // 如果grad_out在CUDA上, 则idx也必须在CUDA上
  if (IS_CUDA_TENSOR(grad_out)) {
    CHECK_CUDA(idx);
  }

  // 创建输出张量, 初始化为0, 形状为[B, C, N]
  at::Tensor output =
      torch::zeros({grad_out.size(0), grad_out.size(1), n},
                   at::device(grad_out.device()).dtype(at::ScalarType::Float));

  // 调用CUDA核函数进行特征采样的反向传播
  if (IS_CUDA_TENSOR(grad_out)) {
    gather_points_grad_kernel_wrapper(grad_out.size(0), grad_out.size(1), n,
                                      idx.size(1), TENSOR_DATA_PTR(grad_out, float),
                                      TENSOR_DATA_PTR(idx, int), TENSOR_DATA_PTR(output, float));
  } else {
    // 仅支持CUDA实现, CPU暂不支持
    AT_CHECK(false, "CPU not supported");
  }

  // 返回输入特征的梯度
  return output;
}

/*
 * @brief 最远点采样主函数(PyTorch接口)
 * 
 * 在输入点云points中, 按照最远点策略采样出nsamples个点的索引, 保证采样点分布均匀。
 * 常用于点云下采样, 覆盖整个点云空间。
 * 
 * @param points   (Tensor) 输入点云坐标, 形状为[B, N, 3]
 * @param nsamples (int)    需要采样的点数
 * @return         (Tensor) 采样点的索引, 形状为[B, nsamples]
 */
at::Tensor furthest_point_sampling(at::Tensor points, const int nsamples) {
  // 检查输入张量是否为连续内存且类型正确
  CHECK_CONTIGUOUS(points);
  CHECK_IS_FLOAT(points);

  // 创建输出张量, 存储采样点索引, 形状为[B, nsamples]
  at::Tensor output =
      torch::zeros({points.size(0), nsamples},
                   at::device(points.device()).dtype(at::ScalarType::Int));

  // 创建临时距离缓存, 初始化为较大值, 形状为[B, N]
  at::Tensor tmp =
      torch::full({points.size(0), points.size(1)}, 1e10,
                  at::device(points.device()).dtype(at::ScalarType::Float));

  // 调用CUDA核函数进行最远点采样
  if (IS_CUDA_TENSOR(points)) {
    furthest_point_sampling_kernel_wrapper(
        points.size(0), points.size(1), nsamples, TENSOR_DATA_PTR(points, float),
        TENSOR_DATA_PTR(tmp, float), TENSOR_DATA_PTR(output, int));
  } else {
    // 仅支持CUDA实现, CPU暂不支持
    AT_CHECK(false, "CPU not supported");
  }

  // 返回采样点索引
  return output;
}
