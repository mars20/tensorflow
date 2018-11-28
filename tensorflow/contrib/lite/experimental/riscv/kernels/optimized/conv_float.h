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
#ifndef TENSORFLOW_CONTRIB_LITE_EXPERIMENTAL_RISCV_KERNELS_OPTIMIZED_CONV_FLOAT_H_
#define TENSORFLOW_CONTRIB_LITE_EXPERIMENTAL_RISCV_KERNELS_OPTIMIZED_CONV_FLOAT_H_

#include "tensorflow/contrib/lite/experimental/riscv/kernels/common.h"
#include "tensorflow/contrib/lite/kernels/internal/types.h"
#include "tensorflow/contrib/lite/experimental/riscv/kernels/optimized/intrinsic/riscv_ml_extension.h"
#include "tensorflow/contrib/lite/experimental/riscv/kernels/optimized/optimized_ops_float.h"
#include <algorithm>

namespace tflite {
namespace optimized_ops {

#ifdef RISCV

inline void Conv(const ConvParams& params, const RuntimeShape& input_shape,
                       const float* input_data, const RuntimeShape& filter_shape,
                       const float* filter_data, const RuntimeShape& bias_shape,
                       const float* bias_data, const RuntimeShape& output_shape,
                       float* output_data, const RuntimeShape& im2col_shape,
                       float* im2col_data) {
  const int stride_width = params.stride_width;
  const int stride_height = params.stride_height;
  const int dilation_width_factor = params.dilation_width_factor;
  const int dilation_height_factor = params.dilation_height_factor;
  const float output_activation_min = params.float_activation_min;
  const float output_activation_max = params.float_activation_max;
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);

  (void)im2col_data;
  (void)im2col_shape;
  // gemmlowp::ScopedProfilingLabel label("Conv");

  // NB: static_cast<float>(0x00000000h) == 0.0f
  const uint8 float_zero_byte = 0x00;
  const float* gemm_input_data = nullptr;
  const RuntimeShape* gemm_input_shape = nullptr;
  const int filter_width = filter_shape.Dims(2);
  const int filter_height = filter_shape.Dims(1);
  const bool need_dilated_im2col =
      dilation_width_factor != 1 || dilation_height_factor != 1;
  const bool need_im2col = stride_width != 1 || stride_height != 1 ||
                           filter_width != 1 || filter_height != 1;
  //  const bool use_kernel1x1 = stride_width == 1 && stride_height == 1 &&
  //                       filter_width == 1 && filter_height == 1;
  if (need_dilated_im2col) {
    DilatedIm2col(params, float_zero_byte, input_shape, input_data,
                  filter_shape, output_shape, im2col_data);
    gemm_input_data = im2col_data;
    gemm_input_shape = &im2col_shape;
  } else if (need_im2col) {
    TFLITE_DCHECK(im2col_data);
    Im2col(params, filter_height, filter_width, float_zero_byte, input_shape,
           input_data, im2col_shape, im2col_data);
    gemm_input_data = im2col_data;
    gemm_input_shape = &im2col_shape;
  } else {
    // TODO(aselle): We need to make sure to not send im2col if it is not
    // needed.
    TFLITE_DCHECK(!im2col_data);
    gemm_input_data = input_data;
    gemm_input_shape = &input_shape;
  }

  // const Dims<4>& gemm_input_dims = ToRuntimeDims(*gemm_input_shape);
  // const Dims<4>& output_dims = ToRuntimeDims(output_shape);
  // const Dims<4>& filter_dims = ToRuntimeDims(filter_shape);
  // const Dims<4>& bias_dims = ToRuntimeDims(bias_shape);

  // const int output_height = ArraySize(output_dims, 2);
  // const int output_width = ArraySize(output_dims, 1);

  // const int batches = MatchingArraySize(gemm_input_dims, 3, output_dims, 3);

  // printf("Input shape: %d %d %d %d\n", gemm_input_shape->Dims(0), gemm_input_shape->Dims(1),
  //        gemm_input_shape->Dims(2), gemm_input_shape->Dims(3));

  // printf("Filter shape: %d %d %d %d\n", filter_shape.Dims(0), filter_shape.Dims(1),
  //        filter_shape.Dims(2), filter_shape.Dims(3));

  // printf("Output shape: %d %d %d %d\n", output_shape.Dims(0), output_shape.Dims(1),
  //        output_shape.Dims(2), output_shape.Dims(3));

  const int input_depth = gemm_input_shape->Dims(3);
 // MatchingArraySize(*gemm_input_dims, 3, filter_dims, 0);
  const int output_depth = MatchingDim(filter_shape, 0, output_shape, 3);

  if (bias_data) {
    TFLITE_DCHECK_EQ(bias_shape.FlatSize(), output_shape.Dims(3));
  }

  const int input_rows = FlatSizeSkipDim(*gemm_input_shape, 3);
  const int output_rows = FlatSizeSkipDim(output_shape, 3);
  const int filter_cols = FlatSizeSkipDim(filter_shape, 0);

  TFLITE_DCHECK_EQ(input_rows, output_rows);
  TFLITE_DCHECK_EQ(input_depth, filter_cols);

  // Number of blocks we can handle based on output depth
  // MAX_BLOCK_SIZE is 4
  int blocks = std::max(std::min(output_depth >> kMaxVectorLength32ShiftOffset, MAX_BLOCK_SIZE), 1);

  // output depth multiplier as function of Maximum Vector length for 32bit Elemenent width
  int output_depth_multiplier = output_depth >> kMaxVectorLength32ShiftOffset;
  int output_depth_remainder = output_depth & (kMaxVectorLength32 - 1);
  bool have_remainder = output_depth_remainder !=0 ? true: false;

  switch(blocks){
    case 1:
      //  printf("Block size of 1\n");
      if(have_remainder && output_depth_multiplier) {
        MatrixMatrixMultiplyAccumulate4x1(gemm_input_data,
                                          input_rows,
                                          input_depth,
                                          filter_data,
                                          output_depth,
                                          filter_cols,
                                          output_data);
        MatrixMatrixMultiplyAccumulate4x1(gemm_input_data,
                                          input_rows,
                                          input_depth,
                                          filter_data + kMaxVectorLength32*filter_cols,
                                          output_depth_remainder,
                                          filter_cols,
                                          output_data + kMaxVectorLength32);
      } else {
        //printf("Block size of 1 and no remainder\n");
        MatrixMatrixMultiplyAccumulate4x1(gemm_input_data,
                                          input_rows,
                                          input_depth,
                                          filter_data,
                                          output_depth,
                                          filter_cols,
                                          output_data);
      }
      break;
    case 2:
      //printf("Block size of 2\n");
      if(have_remainder) {
        MatrixMatrixMultiplyAccumulate4x2(gemm_input_data,
                                          input_rows,
                                          input_depth,
                                          filter_data,
                                          output_depth,
                                          filter_cols,
                                          output_data);
        MatrixMatrixMultiplyAccumulate4x1(gemm_input_data,
                                          input_rows,
                                          input_depth,
                                          filter_data + output_depth_multiplier*kMaxVectorLength32*filter_cols,
                                          output_depth_remainder,
                                          filter_cols,
                                          output_data + output_depth_multiplier*kMaxVectorLength32);
      } else {
        //  printf("Block size of 2 and no remainder\n");
        MatrixMatrixMultiplyAccumulate4x2(gemm_input_data,
                                          input_rows,
                                          input_depth,
                                          filter_data,
                                          output_depth,
                                          filter_cols,
                                          output_data);
      }
      break;
    case 3:
      // printf("Block size of 3\n");
      if(have_remainder) {
        MatrixMatrixMultiplyAccumulate4x3(gemm_input_data,
                                          input_rows,
                                          input_depth,
                                          filter_data,
                                          output_depth,
                                          filter_cols,
                                          output_data);
        MatrixMatrixMultiplyAccumulate4x1(gemm_input_data,
                                          input_rows,
                                          input_depth,
                                          filter_data + output_depth_multiplier*kMaxVectorLength32*filter_cols,
                                          output_depth_remainder,
                                          filter_cols,
                                          output_data + output_depth_multiplier*kMaxVectorLength32);
      } else {
        //  printf("Block size of 3 and no remainder\n");
        MatrixMatrixMultiplyAccumulate4x3(gemm_input_data,
                                          input_rows,
                                          input_depth,
                                          filter_data,
                                          output_depth,
                                          filter_cols,
                                          output_data);
      }
      break;
    case 4:
      //printf("Block size of 4\n");
      if(have_remainder) {
        MatrixMatrixMultiplyAccumulate(gemm_input_data,
                                          input_rows,
                                          input_depth,
                                          filter_data,
                                          output_depth,
                                          filter_cols,
                                          output_data);
        MatrixMatrixMultiplyAccumulate4x1(gemm_input_data,
                                          input_rows,
                                          input_depth,
                                          filter_data + output_depth_multiplier*kMaxVectorLength32*filter_cols,
                                          output_depth_remainder,
                                          filter_cols,
                                          output_data + output_depth_multiplier*kMaxVectorLength32);
      } else {
        // printf("Block size of 4 and no remainder\n");
        MatrixMatrixMultiplyAccumulate(gemm_input_data,
                                       input_rows,
                                       input_depth,
                                       filter_data,
                                       output_depth,
                                       filter_cols,
                                       output_data);
      }
      break;
    default:
       fprintf(stderr, "Unknown block size %d\n", blocks);
       break;
  }

  if (bias_data) {
    AddBiasActivationFunctionWithMinMax(output_data, bias_data,
                                        output_activation_min, output_activation_max,
                                        output_shape.FlatSize(), output_depth);
  }
}
//   for (int batch = 0; batch < batches; ++batch) {
//     for (int out_y = 0; out_y < output_height; ++out_y) {
//       for (int out_x = 0; out_x < output_width; ++out_x) {
//         const float *input_address = gemm_input_data + out_x * gemm_input_dims.strides[1] +
//                                      out_y * gemm_input_dims.strides[2] +
//                                      batch * gemm_input_dims.strides[3];
//         float *output_address = output_data + out_x * output_dims.strides[1] +
//                                 out_y * output_dims.strides[2] +
//                                 batch * output_dims.strides[3];

//         Kernel1x1MultiplyAccumulate(filter_data, input_depth,
//                                     output_depth, input_address,
//                                     output_address);
//       }
//     }
//   }
//   int flattened_len = output_shape.FlatSize();
//   if (bias_data) {
//     AddBiasActivationFunctionWithMinMax(output_data, bias_data,
//                                         output_activation_min, output_activation_max,
//                                         flattened_len, output_depth);
//   }
// }

// inline void Conv(const ConvParams& params, const RuntimeShape& input_shape,
//                  const float* input_data, const RuntimeShape& filter_shape,
//                  const float* filter_data, const RuntimeShape& bias_shape,
//                  const float* bias_data, const RuntimeShape& output_shape,
//                  float* output_data, const RuntimeShape& im2col_shape,
//                  float* im2col_data) {
//   const int stride_width = params.stride_width;
//   const int stride_height = params.stride_height;
//   const int dilation_width_factor = params.dilation_width_factor;
//   const int dilation_height_factor = params.dilation_height_factor;
//   const int pad_width = params.padding_values.width;
//   const int pad_height = params.padding_values.height;
//   const float output_activation_min = params.float_activation_min;
//   const float output_activation_max = params.float_activation_max;
//   TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
//   TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
//   TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);

//   const Dims<4>& input_dims = ToRuntimeDims(input_shape);
//   const Dims<4>& output_dims = ToRuntimeDims(output_shape);
//   const Dims<4>& filter_dims = ToRuntimeDims(filter_shape);

//   (void)im2col_data;   // only used in optimized code.
//   (void)im2col_shape;  // only used in optimized code.
//   const int batches = MatchingDim(input_shape, 0, output_shape, 0);
//   const int input_depth = MatchingDim(input_shape, 3, filter_shape, 3);
//   const int output_depth = MatchingDim(filter_shape, 0, output_shape, 3);
//   if (bias_data) {
//     TFLITE_DCHECK_EQ(bias_shape.FlatSize(), output_depth);
//   }
//   const int input_height = input_shape.Dims(1);
//   const int input_width = input_shape.Dims(2);
//   const int filter_height = filter_shape.Dims(1);
//   const int filter_width = filter_shape.Dims(2);
//   const int output_height = output_shape.Dims(1);
//   const int output_width = output_shape.Dims(2);
//   float temp_ = 0.0;
//   for (int batch = 0; batch < batches; ++batch) {
//     for (int out_y = 0; out_y < output_height; ++out_y) {
//       for (int out_x = 0; out_x < output_width; ++out_x) {
//         for (int out_channel = 0; out_channel < output_depth; ++out_channel) {
//           const int in_x_origin = (out_x * stride_width) - pad_width;
//           const int in_y_origin = (out_y * stride_height) - pad_height;
//           float total = 0.f;
//           for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
//             for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
//               const int in_x = in_x_origin + dilation_width_factor * filter_x;
//               const int in_y =
//                   in_y_origin + dilation_height_factor * filter_y;
//               // If the location is outside the bounds of the input image,
//               // use zero as a default value.
//               if ((in_x >= 0) && (in_x < input_width) && (in_y >= 0) &&
//                   (in_y < input_height)) {
//                 const float* input_address =
//                     input_data + Offset(input_dims, 0, in_x, in_y, batch);
//                 const float* filter_address =
//                     filter_data +
//                     Offset(filter_dims, 0, filter_x, filter_y, out_channel);
//                 VectorVectorMultiplyAccumulate(input_address, filter_address,
//                                                &temp_, input_depth);
//                 total += temp_;
//               }
//             }
//           }
//           float bias_value = 0.0f;
//           if (bias_data) {
//             bias_value = bias_data[out_channel];
//           }
//           output_data[Offset(output_dims, out_channel, out_x, out_y, batch)]
//               = ActivationFunctionWithMinMax(total + bias_value,
//                                            output_activation_min,
//                                            output_activation_max);
//         }
//       }
//     }
//   }
// }
#endif


}  // namespace optimized_ops
}  // namespace tflite

#endif  // TENSORFLOW_CONTRIB_LITE_EXPERIMENTAL_RISCV_KERNELS_OPTIMIZED_FULLY_CONNECTED_FLOAT_H_
