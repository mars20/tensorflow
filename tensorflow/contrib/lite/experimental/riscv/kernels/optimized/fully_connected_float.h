/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#ifndef TENSORFLOW_CONTRIB_LITE_EXPERIMENTAL_RISCV_KERNELS_OPTIMIZED_FULLY_CONNECTED_FLOAT_H_
#define TENSORFLOW_CONTRIB_LITE_EXPERIMENTAL_RISCV_KERNELS_OPTIMIZED_FULLY_CONNECTED_FLOAT_H_

#include "tensorflow/contrib/lite/experimental/riscv/kernels/common.h"
#include "tensorflow/contrib/lite/kernels/internal/types.h"
#include "tensorflow/contrib/lite/experimental/riscv/kernels/optimized/intrinsic/riscv_ml_extension.h"
#include "tensorflow/contrib/lite/experimental/riscv/kernels/optimized/optimized_ops_float.h"

namespace tflite {
namespace optimized_ops {

inline void FullyConnected(
    const FullyConnectedParams& params, const RuntimeShape& input_shape,
    const float* input_data, const RuntimeShape& weights_shape,
    const float* weights_data, const RuntimeShape& bias_shape,
    const float* bias_data, const RuntimeShape& output_shape,
    float* output_data) {
  const float output_activation_min = params.float_activation_min;
  const float output_activation_max = params.float_activation_max;
  // TODO(benoitjacob): This really should be:
  //     const int batches = ArraySize(output_dims, 1);
  // but the current --variable_batch hack consists in overwriting the 3rd
  // dimension with the runtime batch size, as we don't keep track for each
  // array of which dimension is the batch dimension in it.
  const int output_dims_count = output_shape.DimensionsCount();
  const int weights_dims_count = weights_shape.DimensionsCount();
  const int batches = FlatSizeSkipDim(output_shape, output_dims_count - 1);
  const int output_depth = MatchingDim(weights_shape, weights_dims_count - 2,
                                       output_shape, output_dims_count - 1);
  const int accum_depth = weights_shape.Dims(weights_dims_count - 1);
  for (int b = 0; b < batches; ++b) {
    MatrixVectorMultiplyAccumulate(weights_data, output_depth,
                                   accum_depth, input_data,
                                   output_data);
    VectorVectorAddMinMax(output_data, bias_data, output_activation_min,
                          output_activation_max, output_data, output_depth);
  }
}

}  // namespace optimized_ops
}  // namespace tflite

#endif  // TENSORFLOW_CONTRIB_LITE_EXPERIMENTAL_RISCV_KERNELS_OPTIMIZED_FULLY_CONNECTED_FLOAT_H_
