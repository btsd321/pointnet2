// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#ifndef STR
#define STR(x) STR_HELPER(x)
#endif
#ifndef STR_HELPER
#define STR_HELPER(x) #x
#endif

#pragma message("TORCH_VERSION_MAJOR = " STR(TORCH_VERSION_MAJOR))

#if defined(TORCH_VERSION_MAJOR) && TORCH_VERSION_MAJOR >= 1
#define CHECK_MACRO TORCH_CHECK
#else
#define CHECK_MACRO AT_CHECK
#endif

/*
 * @brief 检查输入张量是否为CUDA张量的宏
 *
 * 用于断言x必须是CUDA类型的张量, 否则报错。
 * 常用于自定义CUDA算子的输入检查。
 *
 * @param x (Tensor) 需要检查的输入张量
 */
#define CHECK_CUDA(x)                                                 \
    do                                                                \
    {                                                                 \
        CHECK_MACRO(x.type().is_cuda(), #x " must be a CUDA tensor"); \
    } while (0)

/*
 * @brief 检查输入张量是否为连续内存的宏
 *
 * 用于断言x必须是内存连续(contiguous)的张量, 否则报错。
 * 常用于CUDA算子输入, 保证数据排布正确。
 *
 * @param x (Tensor) 需要检查的输入张量
 */
#define CHECK_CONTIGUOUS(x)                                                \
    do                                                                     \
    {                                                                      \
        CHECK_MACRO(x.is_contiguous(), #x " must be a contiguous tensor"); \
    } while (0)

/*
 * @brief 检查输入张量是否为int类型的宏
 *
 * 用于断言x必须是int类型的张量, 否则报错。
 * 常用于索引等需要整型张量的CUDA算子输入检查。
 *
 * @param x (Tensor) 需要检查的输入张量
 */
#define CHECK_IS_INT(x)                                     \
    do                                                      \
    {                                                       \
        CHECK_MACRO(x.scalar_type() == at::ScalarType::Int, \
                    #x " must be an int tensor");           \
    } while (0)

/*
 * @brief 检查输入张量是否为float类型的宏
 *
 * 用于断言x必须是float类型的张量, 否则报错。
 * 常用于特征、坐标等需要浮点型张量的CUDA算子输入检查。
 *
 * @param x (Tensor) 需要检查的输入张量
 */
#define CHECK_IS_FLOAT(x)                                     \
    do                                                        \
    {                                                         \
        CHECK_MACRO(x.scalar_type() == at::ScalarType::Float, \
                    #x " must be a float tensor");            \      
    } while (0)