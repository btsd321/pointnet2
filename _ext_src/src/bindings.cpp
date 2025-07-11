// Copyright (c) Facebook, Inc. and its affiliates.
// 
// 本源码遵循MIT协议, 详见根目录下的LICENSE文件。

#include "ball_query.h"
#include "group_points.h"
#include "interpolate.h"
#include "sampling.h"

/*
 * @brief PyTorch扩展模块绑定
 * 
 * 使用pybind11将C++/CUDA实现的点云操作函数注册为Python可调用接口。
 * 每个m.def对应一个Python可调用函数, 便于在PyTorch中直接调用高效的底层实现。
 */
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // @brief 点云特征采样(Gather Points)
  // 根据索引采样点云特征, 返回采样后的特征
  m.def("gather_points", &gather_points);

  // @brief 点云特征采样的反向传播
  // 计算gather_points操作的梯度, 将梯度累加回原始特征
  m.def("gather_points_grad", &gather_points_grad);

  // @brief 最远点采样(Furthest Point Sampling, FPS)
  // 按最远点策略采样点云, 返回采样点索引
  m.def("furthest_point_sampling", &furthest_point_sampling);

  // @brief 三近邻查找(Three Nearest Neighbors)
  // 查找每个点最近的3个已知点, 返回索引和距离
  m.def("three_nn", &three_nn);

  // @brief 三线性插值(Three Interpolate)
  // 根据三近邻索引和权重, 对特征进行插值
  m.def("three_interpolate", &three_interpolate);

  // @brief 三线性插值的反向传播
  // 计算插值操作的梯度, 将梯度累加回原始特征
  m.def("three_interpolate_grad", &three_interpolate_grad);

  // @brief 球查询(Ball Query)
  // 以中心点为球心, 查找半径内的邻域点索引
  m.def("ball_query", &ball_query);

  // @brief 点云分组操作(Group Points)
  // 根据索引分组采样点云特征, 返回分组特征
  m.def("group_points", &group_points);

  // @brief 点云分组操作的反向传播
  // 计算group_points操作的梯度, 将梯度累加回原始特征
  m.def("group_points_grad", &group_points_grad);
}
