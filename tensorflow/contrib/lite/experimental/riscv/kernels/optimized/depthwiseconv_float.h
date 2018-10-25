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
#ifndef TENSORFLOW_CONTRIB_LITE_EXPERIMENTAL_RISCV_KERNELS_OPTIMIZED_DEPTHWISECONV_FLOAT_H_
#define TENSORFLOW_CONTRIB_LITE_EXPERIMENTAL_RISCV_KERNELS_OPTIMIZED_DEPTHWISECONV_FLOAT_H_

#include "tensorflow/contrib/lite/experimental/riscv/kernels/optimized/intrinsic/riscv_ml_extension.h"
#include "tensorflow/contrib/lite/experimental/riscv/kernels/common.h"
#include "tensorflow/contrib/lite/kernels/internal/compatibility.h"
#include "tensorflow/contrib/lite/kernels/internal/types.h"

namespace tflite {
namespace optimized_ops {

#ifdef RISCV
inline void DepthwiseConv(
    const DepthwiseParams& params, const RuntimeShape& input_shape,
    const float* input_data, const RuntimeShape& filter_shape,
    const float* filter_data, const RuntimeShape& bias_shape,
    const float* bias_data, const RuntimeShape& output_shape,
    float* output_data) {
  const int stride_width = params.stride_width;
  const int stride_height = params.stride_height;
  const int dilation_width_factor = params.dilation_width_factor;
  const int dilation_height_factor = params.dilation_height_factor;
  const int pad_width = params.padding_values.width;
  const int pad_height = params.padding_values.height;
  const int depth_multiplier = params.depth_multiplier;
  const float output_activation_min = params.float_activation_min;
  const float output_activation_max = params.float_activation_max;
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);

  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int output_depth = MatchingDim(filter_shape, 3, output_shape, 3);
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int input_depth = input_shape.Dims(3);
  const int filter_height = filter_shape.Dims(1);
  const int filter_width = filter_shape.Dims(2);
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);
  bool use_bias_first = false;
  const Dims<4>& input_dims = ToRuntimeDims(input_shape);
  const Dims<4>& output_dims = ToRuntimeDims(output_shape);
  const Dims<4>& filter_dims = ToRuntimeDims(filter_shape);

  TFLITE_DCHECK_EQ(output_depth, input_depth * depth_multiplier);
  TFLITE_DCHECK_EQ(bias_shape.FlatSize(), output_depth);

  for (int b = 0; b < batches; ++b) {
    for (int out_y = 0; out_y < output_height; ++out_y) {
      for (int out_x = 0; out_x < output_width; ++out_x) {
        const int in_y_origin = (out_y * stride_height) - pad_height;
        const int in_x_origin = (out_x * stride_width) - pad_width;
        const int filter_y_start = std::max(0, -in_y_origin);
        const int filter_y_end =
            std::min(filter_height, input_height - in_y_origin);
        const int filter_x_start = std::max(0, -in_x_origin);
        const int filter_x_end =
            std::min(filter_width, input_width - in_x_origin);
        if (bias_data) {
          use_bias_first = true;
        }
        float* output_address =
                &output_data[Offset(output_dims, 0, out_x, out_y, b)];
        for (int filter_y = 0; filter_y < filter_height;
             ++filter_y) {
          for (int filter_x = 0; filter_x < filter_width;
               ++filter_x) {
            const int in_x = in_x_origin + dilation_width_factor *  filter_x;
            const int in_y = in_y_origin + dilation_height_factor * filter_y;
            if ((in_x >= 0) && (in_x < input_width) && (in_y >= 0) &&
                    (in_y < input_height)) {
              const float* input_address =
                  &input_data[Offset(input_dims, 0, in_x, in_y, b)];
              const float* filter_address =
                  &filter_data[Offset(filter_dims, 0, filter_x, filter_y, 0)];
              VectorVectorMultiplyAccumulateDepthwise(
                  input_address, filter_address, output_address, input_depth,
                  depth_multiplier, bias_data, use_bias_first);
              use_bias_first = false;
            }
          }
        }
        VectorActivationFunctionWithMinMax(output_address, output_activation_min, output_activation_max, output_depth);
      }
    }
  }
}
#endif
}  // end namespace optimized_ops
}  // end namespace tflite

#endif  // TENSORFLOW_CONTRIB_LITE_EXPERIMENTAL_RISCV_KERNELS_OPTIMIZED_DEPTHWISECONV_FLOAT_H_
