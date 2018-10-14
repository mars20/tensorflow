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
#ifndef TENSORFLOW_CONTRIB_LITE_EXPERIMENTAL_RISCV_KERNELS_REFERENCE_REFERENCE_OPS_FLOAT_H_
#define TENSORFLOW_CONTRIB_LITE_EXPERIMENTAL_RISCV_KERNELS_REFERENCE_REFERENCE_OPS_FLOAT_H_

#include <stdint.h>
#include <sys/types.h>
#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <memory>
#include <type_traits>

#include "tensorflow/contrib/lite/experimental/riscv/kernels/common.h"
#include "tensorflow/contrib/lite/kernels/internal/strided_slice_logic.h"
#include "tensorflow/contrib/lite/kernels/internal/types.h"

namespace tflite {
namespace reference_ops {

template <typename T>
inline void DepthToSpace(const tflite::DepthToSpaceParams& op_params,
                         const RuntimeShape& unextended_input_shape,
                         const T* input_data,
                         const RuntimeShape& unextended_output_shape,
                         T* output_data) {
  TFLITE_DCHECK_LE(unextended_input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_LE(unextended_output_shape.DimensionsCount(), 4);
  const RuntimeShape input_shape =
      RuntimeShape::ExtendedShape(4, unextended_input_shape);
  const RuntimeShape output_shape =
      RuntimeShape::ExtendedShape(4, unextended_output_shape);

  const int input_depth = input_shape.Dims(3);
  const int input_width = input_shape.Dims(2);
  const int input_height = input_shape.Dims(1);
  const int input_batch = input_shape.Dims(0);

  const int output_depth = output_shape.Dims(3);
  const int output_width = output_shape.Dims(2);
  const int output_height = output_shape.Dims(1);
  const int output_batch = output_shape.Dims(0);

  const int32 block_size = op_params.block_size;

  TFLITE_DCHECK_EQ(input_width * block_size, output_width);
  TFLITE_DCHECK_EQ(input_height * block_size, output_height);
  TFLITE_DCHECK_EQ(input_depth, output_depth * block_size * block_size);
  TFLITE_DCHECK_EQ(input_batch, output_batch);

  for (int out_b = 0; out_b < output_batch; ++out_b) {
    for (int out_h = 0; out_h < output_height; ++out_h) {
      for (int out_w = 0; out_w < output_width; ++out_w) {
        for (int out_d = 0; out_d < output_depth; ++out_d) {
          const int in_d =
              out_d + ((out_h % block_size) * block_size + out_w % block_size) *
                          output_depth;

          const int in_w = out_w / block_size;
          const int in_h = out_h / block_size;
          const int in_b = out_b;

          const int input_index = Offset(input_shape, in_b, in_h, in_w, in_d);
          const int output_index =
              Offset(output_shape, out_b, out_h, out_w, out_d);

          output_data[output_index] = input_data[input_index];
        }
      }
    }
  }
}

template <typename T>
inline void SpaceToDepth(const tflite::SpaceToDepthParams& op_params,
                         const RuntimeShape& unextended_input_shape,
                         const T* input_data,
                         const RuntimeShape& unextended_output_shape,
                         T* output_data) {
  TFLITE_DCHECK_LE(unextended_input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_LE(unextended_output_shape.DimensionsCount(), 4);
  const RuntimeShape input_shape =
      RuntimeShape::ExtendedShape(4, unextended_input_shape);
  const RuntimeShape output_shape =
      RuntimeShape::ExtendedShape(4, unextended_output_shape);

  const int input_depth = input_shape.Dims(3);
  const int input_width = input_shape.Dims(2);
  const int input_height = input_shape.Dims(1);
  const int input_batch = input_shape.Dims(0);

  const int output_depth = output_shape.Dims(3);
  const int output_width = output_shape.Dims(2);
  const int output_height = output_shape.Dims(1);
  const int output_batch = output_shape.Dims(0);

  const int32 block_size = op_params.block_size;

  TFLITE_DCHECK_EQ(input_width, output_width * block_size);
  TFLITE_DCHECK_EQ(input_height, output_height * block_size);
  TFLITE_DCHECK_EQ(input_depth * block_size * block_size, output_depth);
  TFLITE_DCHECK_EQ(input_batch, output_batch);

  for (int in_b = 0; in_b < input_batch; ++in_b) {
    for (int in_h = 0; in_h < input_height; ++in_h) {
      for (int in_w = 0; in_w < input_width; ++in_w) {
        for (int in_d = 0; in_d < input_depth; ++in_d) {
          const int out_d =
              in_d + ((in_h % block_size) * block_size + in_w % block_size) *
                         input_depth;
          const int out_w = in_w / block_size;
          const int out_h = in_h / block_size;
          const int out_b = in_b;

          const int input_index = Offset(input_shape, in_b, in_h, in_w, in_d);
          const int output_index =
              Offset(output_shape, out_b, out_h, out_w, out_d);

          output_data[output_index] = input_data[input_index];
        }
      }
    }
  }
}

inline void L2Normalization(const tflite::L2NormalizationParams& op_params,
                            const RuntimeShape& input_shape,
                            const float* input_data,
                            const RuntimeShape& output_shape,
                            float* output_data) {
  const int trailing_dim = input_shape.DimensionsCount() - 1;
  const int outer_size =
      MatchingFlatSizeSkipDim(input_shape, trailing_dim, output_shape);
  const int depth =
      MatchingDim(input_shape, trailing_dim, output_shape, trailing_dim);
  for (int i = 0; i < outer_size; ++i) {
    float squared_l2_norm = 0;
    for (int c = 0; c < depth; ++c) {
      const float val = input_data[depth * i + c];
      squared_l2_norm += val * val;
    }
    const float l2_norm = std::sqrt(squared_l2_norm);
    for (int c = 0; c < depth; ++c) {
      output_data[depth * i + c] = input_data[depth * i + c] / l2_norm;
    }
  }
}

inline void Add(const ArithmeticParams& params,
                const RuntimeShape& input1_shape, const float* input1_data,
                const RuntimeShape& input2_shape, const float* input2_data,
                const RuntimeShape& output_shape, float* output_data) {
  const int size = MatchingFlatSize(input1_shape, input2_shape, output_shape);
  for (int i = 0; i < size; i++) {
    auto x = input1_data[i] + input2_data[i];
    output_data[i] = ActivationFunctionWithMinMax(
        x, params.float_activation_min, params.float_activation_max);
  }
}

// TODO(jiawen): We can implement BroadcastAdd on buffers of arbitrary
// dimensionality if the runtime code does a single loop over one dimension
// that handles broadcasting as the base case. The code generator would then
// generate max(D1, D2) nested for loops.
// TODO(benoitjacob): BroadcastAdd is intentionally duplicated from
// reference_ops.h. Once an optimized version is implemented and NdArrayDesc<T>
// is no longer referenced in this file, move NdArrayDesc<T> from types.h to
// reference_ops.h.
inline void BroadcastAdd4DSlow(const ArithmeticParams& params,
                               const RuntimeShape& input1_shape,
                               const float* input1_data,
                               const RuntimeShape& input2_shape,
                               const float* input2_data,
                               const RuntimeShape& output_shape,
                               float* output_data) {
  NdArrayDesc<4> desc1;
  NdArrayDesc<4> desc2;
  NdArrayDescsForElementwiseBroadcast(input1_shape, input2_shape, &desc1,
                                      &desc2);
  const RuntimeShape extended_output_shape =
      RuntimeShape::ExtendedShape(4, output_shape);

  // In Tensorflow, the dimensions are canonically named (batch_number, row,
  // col, channel), with extents (batches, height, width, depth), with the
  // trailing dimension changing most rapidly (channels has the smallest stride,
  // typically 1 element).
  //
  // In generated C code, we store arrays with the dimensions reversed. The
  // first dimension has smallest stride.
  //
  // We name our variables by their Tensorflow convention, but generate C code
  // nesting loops such that the innermost loop has the smallest stride for the
  // best cache behavior.
  for (int b = 0; b < extended_output_shape.Dims(0); ++b) {
    for (int y = 0; y < extended_output_shape.Dims(1); ++y) {
      for (int x = 0; x < extended_output_shape.Dims(2); ++x) {
        for (int c = 0; c < extended_output_shape.Dims(3); ++c) {
          output_data[Offset(extended_output_shape, b, y, x, c)] =
              ActivationFunctionWithMinMax(
                  input1_data[SubscriptToIndex(desc1, b, y, x, c)] +
                      input2_data[SubscriptToIndex(desc2, b, y, x, c)],
                  params.float_activation_min, params.float_activation_max);
        }
      }
    }
  }
}

template <typename T>
inline void Mul(const ArithmeticParams& params,
                const RuntimeShape& input1_shape, const T* input1_data,
                const RuntimeShape& input2_shape, const T* input2_data,
                const RuntimeShape& output_shape, T* output_data) {
  T output_activation_min;
  T output_activation_max;
  GetActivationParams(params, &output_activation_min, &output_activation_max);

  const int flat_size =
      MatchingFlatSize(input1_shape, input2_shape, output_shape);
  for (int i = 0; i < flat_size; ++i) {
    output_data[i] = ActivationFunctionWithMinMax(
        input1_data[i] * input2_data[i], output_activation_min,
        output_activation_max);
  }
}

// TODO(jiawen): We can implement BroadcastMul on buffers of arbitrary
// dimensionality if the runtime code does a single loop over one dimension
// that handles broadcasting as the base case. The code generator would then
// generate max(D1, D2) nested for loops.
// TODO(benoitjacob): BroadcastMul is intentionally duplicated from
// reference_ops.h. Once an optimized version is implemented and NdArrayDesc<T>
// is no longer referenced in this file, move NdArrayDesc<T> from types.h to
// reference_ops.h.
template <typename T>
void BroadcastMul4DSlow(const ArithmeticParams& params,
                        const RuntimeShape& unextended_input1_shape,
                        const T* input1_data,
                        const RuntimeShape& unextended_input2_shape,
                        const T* input2_data,
                        const RuntimeShape& unextended_output_shape,
                        T* output_data) {
  T output_activation_min;
  T output_activation_max;
  GetActivationParams(params, &output_activation_min, &output_activation_max);

  TFLITE_DCHECK_LE(unextended_input1_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_LE(unextended_input2_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_LE(unextended_output_shape.DimensionsCount(), 4);
  const RuntimeShape output_shape =
      RuntimeShape::ExtendedShape(4, unextended_output_shape);

  NdArrayDesc<4> desc1;
  NdArrayDesc<4> desc2;
  NdArrayDescsForElementwiseBroadcast(unextended_input1_shape,
                                      unextended_input2_shape, &desc1, &desc2);

  // In Tensorflow, the dimensions are canonically named (batch_number, row,
  // col, channel), with extents (batches, height, width, depth), with the
  // trailing dimension changing most rapidly (channels has the smallest stride,
  // typically 1 element).
  //
  // In generated C code, we store arrays with the dimensions reversed. The
  // first dimension has smallest stride.
  //
  // We name our variables by their Tensorflow convention, but generate C code
  // nesting loops such that the innermost loop has the smallest stride for the
  // best cache behavior.
  for (int b = 0; b < output_shape.Dims(0); ++b) {
    for (int y = 0; y < output_shape.Dims(1); ++y) {
      for (int x = 0; x < output_shape.Dims(2); ++x) {
        for (int c = 0; c < output_shape.Dims(3); ++c) {
          output_data[Offset(output_shape, b, y, x, c)] =
              ActivationFunctionWithMinMax(
                  input1_data[SubscriptToIndex(desc1, b, y, x, c)] *
                      input2_data[SubscriptToIndex(desc2, b, y, x, c)],
                  output_activation_min, output_activation_max);
        }
      }
    }
  }
}

template <typename T>
inline void Div(const ArithmeticParams& params,
                const RuntimeShape& input1_shape, const T* input1_data,
                const RuntimeShape& input2_shape, const T* input2_data,
                const RuntimeShape& output_shape, T* output_data) {
  T output_activation_min;
  T output_activation_max;
  GetActivationParams(params, &output_activation_min, &output_activation_max);

  const int flat_size =
      MatchingFlatSize(input1_shape, input2_shape, output_shape);
  for (int i = 0; i < flat_size; ++i) {
    output_data[i] = ActivationFunctionWithMinMax(
        input1_data[i] / input2_data[i], output_activation_min,
        output_activation_max);
  }
}

// TODO(jiawen): We can implement BroadcastDiv on buffers of arbitrary
// dimensionality if the runtime code does a single loop over one dimension
// that handles broadcasting as the base case. The code generator would then
// generate max(D1, D2) nested for loops.
template <typename T>
void BroadcastDiv4DSlow(const ArithmeticParams& params,
                        const RuntimeShape& unextended_input1_shape,
                        const T* input1_data,
                        const RuntimeShape& unextended_input2_shape,
                        const T* input2_data,
                        const RuntimeShape& unextended_output_shape,
                        T* output_data) {
  T output_activation_min;
  T output_activation_max;
  GetActivationParams(params, &output_activation_min, &output_activation_max);

  TFLITE_DCHECK_LE(unextended_input1_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_LE(unextended_input2_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_LE(unextended_output_shape.DimensionsCount(), 4);
  const RuntimeShape output_shape =
      RuntimeShape::ExtendedShape(4, unextended_output_shape);

  NdArrayDesc<4> desc1;
  NdArrayDesc<4> desc2;
  NdArrayDescsForElementwiseBroadcast(unextended_input1_shape,
                                      unextended_input2_shape, &desc1, &desc2);

  // In Tensorflow, the dimensions are canonically named (batch_number, row,
  // col, channel), with extents (batches, height, width, depth), with the
  // trailing dimension changing most rapidly (channels has the smallest
  // stride, typically 1 element).
  //
  // In generated C code, we store arrays with the dimensions reversed. The
  // first dimension has smallest stride.
  //
  // We name our variables by their Tensorflow convention, but generate C code
  // nesting loops such that the innermost loop has the smallest stride for
  // the best cache behavior.
  for (int b = 0; b < output_shape.Dims(0); ++b) {
    for (int y = 0; y < output_shape.Dims(1); ++y) {
      for (int x = 0; x < output_shape.Dims(2); ++x) {
        for (int c = 0; c < output_shape.Dims(3); ++c) {
          output_data[Offset(output_shape, b, y, x, c)] =
              ActivationFunctionWithMinMax(
                  input1_data[SubscriptToIndex(desc1, b, y, x, c)] /
                      input2_data[SubscriptToIndex(desc2, b, y, x, c)],
                  output_activation_min, output_activation_max);
        }
      }
    }
  }
}

inline void SubNonBroadcast(const ArithmeticParams& params,
                            const RuntimeShape& input1_shape,
                            const float* input1_data,
                            const RuntimeShape& input2_shape,
                            const float* input2_data,
                            const RuntimeShape& output_shape,
                            float* output_data) {
  const int flat_size =
      MatchingFlatSize(input1_shape, input2_shape, output_shape);
  for (int i = 0; i < flat_size; ++i) {
    output_data[i] = ActivationFunctionWithMinMax(
        input1_data[i] - input2_data[i], params.float_activation_min,
        params.float_activation_max);
  }
}

template <typename T>
void Sub(const ArithmeticParams& params, const RuntimeShape& input1_shape,
         const T* input1_data, const RuntimeShape& input2_shape,
         const T* input2_data, const RuntimeShape& output_shape,
         T* output_data) {
  NdArrayDesc<4> desc1;
  NdArrayDesc<4> desc2;
  NdArrayDescsForElementwiseBroadcast(input1_shape, input2_shape, &desc1,
                                      &desc2);
  const RuntimeShape extended_output_shape =
      RuntimeShape::ExtendedShape(4, output_shape);

  // In Tensorflow, the dimensions are canonically named (batch_number, row,
  // col, channel), with extents (batches, height, width, depth), with the
  // trailing dimension changing most rapidly (channels has the smallest stride,
  // typically 1 element).
  //
  // In generated C code, we store arrays with the dimensions reversed. The
  // first dimension has smallest stride.
  //
  // We name our variables by their Tensorflow convention, but generate C code
  // nesting loops such that the innermost loop has the smallest stride for the
  // best cache behavior.
  for (int b = 0; b < extended_output_shape.Dims(0); ++b) {
    for (int y = 0; y < extended_output_shape.Dims(1); ++y) {
      for (int x = 0; x < extended_output_shape.Dims(2); ++x) {
        for (int c = 0; c < extended_output_shape.Dims(3); ++c) {
          output_data[Offset(extended_output_shape, b, y, x, c)] =
              input1_data[SubscriptToIndex(desc1, b, y, x, c)] -
              input2_data[SubscriptToIndex(desc2, b, y, x, c)];
        }
      }
    }
  }
}
// TODO(jiawen): We can implement BroadcastSub on buffers of arbitrary
// dimensionality if the runtime code does a single loop over one dimension
// that handles broadcasting as the base case. The code generator would then
// generate max(D1, D2) nested for loops.
// TODO(benoitjacob): BroadcastSub is intentionally duplicated from
// reference_ops.h. Once an optimized version is implemented and NdArrayDesc<T>
// is no longer referenced in this file, move NdArrayDesc<T> from types.h to
// reference_ops.h.
inline void BroadcastSub4DSlow(const ArithmeticParams& params,
                               const RuntimeShape& input1_shape,
                               const float* input1_data,
                               const RuntimeShape& input2_shape,
                               const float* input2_data,
                               const RuntimeShape& output_shape,
                               float* output_data) {
  NdArrayDesc<4> desc1;
  NdArrayDesc<4> desc2;
  NdArrayDescsForElementwiseBroadcast(input1_shape, input2_shape, &desc1,
                                      &desc2);
  const RuntimeShape extended_output_shape =
      RuntimeShape::ExtendedShape(4, output_shape);

  // In Tensorflow, the dimensions are canonically named (batch_number, row,
  // col, channel), with extents (batches, height, width, depth), with the
  // trailing dimension changing most rapidly (channels has the smallest stride,
  // typically 1 element).
  //
  // In generated C code, we store arrays with the dimensions reversed. The
  // first dimension has smallest stride.
  //
  // We name our variables by their Tensorflow convention, but generate C code
  // nesting loops such that the innermost loop has the smallest stride for the
  // best cache behavior.
  for (int b = 0; b < extended_output_shape.Dims(0); ++b) {
    for (int y = 0; y < extended_output_shape.Dims(1); ++y) {
      for (int x = 0; x < extended_output_shape.Dims(2); ++x) {
        for (int c = 0; c < extended_output_shape.Dims(3); ++c) {
          output_data[Offset(extended_output_shape, b, y, x, c)] =
              ActivationFunctionWithMinMax(
                  input1_data[SubscriptToIndex(desc1, b, y, x, c)] -
                      input2_data[SubscriptToIndex(desc2, b, y, x, c)],
                  params.float_activation_min, params.float_activation_max);
        }
      }
    }
  }
}

inline void SubWithActivation(const ArithmeticParams& params,
                              const RuntimeShape& input1_shape,
                              const float* input1_data,
                              const RuntimeShape& input2_shape,
                              const float* input2_data,
                              const RuntimeShape& output_shape,
                              float* output_data) {
  const int flat_size =
      MatchingFlatSize(input1_shape, input2_shape, input2_shape);
  for (int i = 0; i < flat_size; ++i) {
    output_data[i] = ActivationFunctionWithMinMax(
        input1_data[i] - input2_data[i], params.float_activation_min,
        params.float_activation_max);
  }
}

template <typename Scalar>
inline void Concatenation(const ConcatenationParams& params,
                          const RuntimeShape* const* input_shapes,
                          const Scalar* const* input_data,
                          const RuntimeShape& output_shape,
                          Scalar* output_data) {
  int axis = params.axis;
  int inputs_count = params.inputs_count;
  const int concat_dimensions = output_shape.DimensionsCount();
  TFLITE_DCHECK_LT(axis, concat_dimensions);

  int64_t concat_size = 0;
  for (int i = 0; i < inputs_count; i++) {
    TFLITE_DCHECK_EQ(input_shapes[i]->DimensionsCount(), concat_dimensions);
    for (int j = 0; j < concat_dimensions; j++) {
      if (j != axis) {
        MatchingDim(*input_shapes[i], j, output_shape, j);
      }
    }
    concat_size += input_shapes[i]->Dims(axis);
  }
  TFLITE_DCHECK_EQ(concat_size, output_shape.Dims(axis));
  int64_t outer_size = 1;
  for (int i = 0; i < axis; ++i) {
    outer_size *= output_shape.Dims(i);
  }
  // For all input arrays,
  // FlatSize() = outer_size * Dims(axis) * base_inner_size;
  int64_t base_inner_size = 1;
  for (int i = axis + 1; i < concat_dimensions; ++i) {
    base_inner_size *= output_shape.Dims(i);
  }

  Scalar* output_ptr = output_data;
  for (int k = 0; k < outer_size; k++) {
    for (int i = 0; i < inputs_count; ++i) {
      const int copy_size = input_shapes[i]->Dims(axis) * base_inner_size;
      memcpy(output_ptr, input_data[i] + k * copy_size,
             copy_size * sizeof(Scalar));
      output_ptr += copy_size;
    }
  }
}

template <typename Scalar>
void Pack(const PackParams& params, const RuntimeShape* const* input_shapes,
          const Scalar* const* input_data, const RuntimeShape& output_shape,
          Scalar* output_data) {
  const int dimensions = output_shape.DimensionsCount();
  int axis = params.axis;
  int inputs_count = params.inputs_count;

  int outer_size = 1;
  for (int i = 0; i < axis; i++) {
    outer_size *= output_shape.Dims(i);
  }
  int copy_size = 1;
  for (int i = params.axis + 1; i < dimensions; i++) {
    copy_size *= output_shape.Dims(i);
  }
  TFLITE_DCHECK_EQ((**input_shapes).FlatSize(), copy_size * outer_size);

  for (int i = 0; i < inputs_count; ++i) {
    for (int k = 0; k < outer_size; k++) {
      const Scalar* input_ptr = input_data[i] + copy_size * k;
      int loc = k * inputs_count * copy_size + i * copy_size;
      memcpy(output_data + loc, input_ptr, copy_size * sizeof(Scalar));
    }
  }
}

template <typename Scalar>
void Unpack(const UnpackParams& params, const RuntimeShape& input_shape,
            const Scalar* input_data, const RuntimeShape& output_shape,
            Scalar* const* output_datas) {
  const int dimensions = input_shape.DimensionsCount();
  const int outputs_count = params.num_split;

  int outer_size = 1;
  for (int i = 0; i < params.axis; i++) {
    outer_size *= input_shape.Dims(i);
  }
  int copy_size = 1;
  for (int i = params.axis + 1; i < dimensions; i++) {
    copy_size *= input_shape.Dims(i);
  }
  TFLITE_DCHECK_EQ(output_shape.FlatSize(), copy_size * outer_size);

  for (int i = 0; i < outputs_count; ++i) {
    for (int k = 0; k < outer_size; k++) {
      Scalar* output_ptr = output_datas[i] + copy_size * k;
      int loc = k * outputs_count * copy_size + i * copy_size;
      memcpy(output_ptr, input_data + loc, copy_size * sizeof(Scalar));
    }
  }
}

template <typename Scalar>
void DepthConcatenation(const ConcatenationParams& params,
                        const RuntimeShape* const* input_shapes,
                        const Scalar* const* input_data,
                        const RuntimeShape& output_shape, Scalar* output_data) {
  auto params_copy = params;
  params_copy.axis = 3;
  Concatenation(params_copy, input_shapes, input_data, output_shape,
                output_data);
}

template <typename Scalar>
void Split(const SplitParams& params, const RuntimeShape& input_shape,
           const Scalar* input_data, const RuntimeShape* const* output_shapes,
           Scalar* const* output_data) {
  const int concat_dimensions = input_shape.DimensionsCount();
  int axis = params.axis < 0 ? params.axis + concat_dimensions : params.axis;
  int outputs_count = params.num_split;
  TFLITE_DCHECK_LT(axis, concat_dimensions);

  int64_t concat_size = 0;
  for (int i = 0; i < outputs_count; i++) {
    TFLITE_DCHECK_EQ(output_shapes[i]->DimensionsCount(), concat_dimensions);
    for (int j = 0; j < concat_dimensions; j++) {
      if (j != axis) {
        MatchingDim(*output_shapes[i], j, input_shape, j);
      }
    }
    concat_size += output_shapes[i]->Dims(axis);
  }
  TFLITE_DCHECK_EQ(concat_size, input_shape.Dims(axis));
  int64_t outer_size = 1;
  for (int i = 0; i < axis; ++i) {
    outer_size *= input_shape.Dims(i);
  }
  // For all output arrays,
  // FlatSize() = outer_size * Dims(axis) * base_inner_size;
  int64_t base_inner_size = 1;
  for (int i = axis + 1; i < concat_dimensions; ++i) {
    base_inner_size *= input_shape.Dims(i);
  }

  const Scalar* input_ptr = input_data;
  for (int k = 0; k < outer_size; k++) {
    for (int i = 0; i < outputs_count; ++i) {
      const int copy_size = output_shapes[i]->Dims(axis) * base_inner_size;
      memcpy(output_data[i] + k * copy_size, input_ptr,
             copy_size * sizeof(Scalar));
      input_ptr += copy_size;
    }
  }
}

inline int NodeOffset(int b, int h, int w, int height, int width) {
  return (b * height + h) * width + w;
}

inline void AveragePool(const PoolParams& params,
                        const RuntimeShape& input_shape,
                        const float* input_data,
                        const RuntimeShape& output_shape, float* output_data) {
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);
  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int depth = MatchingDim(input_shape, 3, output_shape, 3);
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);
  const int stride_height = params.stride_height;
  const int stride_width = params.stride_width;
  for (int batch = 0; batch < batches; ++batch) {
    for (int out_y = 0; out_y < output_height; ++out_y) {
      for (int out_x = 0; out_x < output_width; ++out_x) {
        for (int channel = 0; channel < depth; ++channel) {
          const int in_x_origin =
              (out_x * stride_width) - params.padding_values.width;
          const int in_y_origin =
              (out_y * stride_height) - params.padding_values.height;
          // Compute the boundaries of the filter region clamped so as to
          // ensure that the filter window fits in the input array.
          const int filter_x_start = std::max(0, -in_x_origin);
          const int filter_x_end =
              std::min(params.filter_width, input_width - in_x_origin);
          const int filter_y_start = std::max(0, -in_y_origin);
          const int filter_y_end =
              std::min(params.filter_height, input_height - in_y_origin);
          float total = 0.f;
          float filter_count = 0;
          for (int filter_y = filter_y_start; filter_y < filter_y_end;
               ++filter_y) {
            for (int filter_x = filter_x_start; filter_x < filter_x_end;
                 ++filter_x) {
              const int in_x = in_x_origin + filter_x;
              const int in_y = in_y_origin + filter_y;
              total +=
                  input_data[Offset(input_shape, batch, in_y, in_x, channel)];
              filter_count++;
            }
          }
          const float average = total / filter_count;
          output_data[Offset(output_shape, batch, out_y, out_x, channel)] =
              ActivationFunctionWithMinMax(average, params.float_activation_min,
                                           params.float_activation_max);
        }
      }
    }
  }
}

inline void L2Pool(const PoolParams& params, const RuntimeShape& input_shape,
                   const float* input_data, const RuntimeShape& output_shape,
                   float* output_data) {
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);
  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int depth = MatchingDim(input_shape, 3, output_shape, 3);
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);
  const int stride_height = params.stride_height;
  const int stride_width = params.stride_width;
  for (int batch = 0; batch < batches; ++batch) {
    for (int out_y = 0; out_y < output_height; ++out_y) {
      for (int out_x = 0; out_x < output_width; ++out_x) {
        for (int channel = 0; channel < depth; ++channel) {
          const int in_x_origin =
              (out_x * stride_width) - params.padding_values.width;
          const int in_y_origin =
              (out_y * stride_height) - params.padding_values.height;
          // Compute the boundaries of the filter region clamped so as to
          // ensure that the filter window fits in the input array.
          const int filter_x_start = std::max(0, -in_x_origin);
          const int filter_x_end =
              std::min(params.filter_width, input_width - in_x_origin);
          const int filter_y_start = std::max(0, -in_y_origin);
          const int filter_y_end =
              std::min(params.filter_height, input_height - in_y_origin);
          float sum_squares = 0.f;
          int filter_count = 0;
          for (int filter_y = filter_y_start; filter_y < filter_y_end;
               ++filter_y) {
            for (int filter_x = filter_x_start; filter_x < filter_x_end;
                 ++filter_x) {
              const int in_x = in_x_origin + filter_x;
              const int in_y = in_y_origin + filter_y;
              const float val =
                  input_data[Offset(input_shape, batch, in_y, in_x, channel)];
              sum_squares += val * val;
              filter_count++;
            }
          }
          const float l2pool_result = std::sqrt(sum_squares / filter_count);
          output_data[Offset(output_shape, batch, out_y, out_x, channel)] =
              ActivationFunctionWithMinMax(l2pool_result,
                                           params.float_activation_min,
                                           params.float_activation_max);
        }
      }
    }
  }
}

inline void MaxPool(const PoolParams& params, const RuntimeShape& input_shape,
                    const float* input_data, const RuntimeShape& output_shape,
                    float* output_data) {
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);
  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int depth = MatchingDim(input_shape, 3, output_shape, 3);
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);
  const int stride_height = params.stride_height;
  const int stride_width = params.stride_width;
  for (int batch = 0; batch < batches; ++batch) {
    for (int out_y = 0; out_y < output_height; ++out_y) {
      for (int out_x = 0; out_x < output_width; ++out_x) {
        for (int channel = 0; channel < depth; ++channel) {
          const int in_x_origin =
              (out_x * stride_width) - params.padding_values.width;
          const int in_y_origin =
              (out_y * stride_height) - params.padding_values.height;
          // Compute the boundaries of the filter region clamped so as to
          // ensure that the filter window fits in the input array.
          const int filter_x_start = std::max(0, -in_x_origin);
          const int filter_x_end =
              std::min(params.filter_width, input_width - in_x_origin);
          const int filter_y_start = std::max(0, -in_y_origin);
          const int filter_y_end =
              std::min(params.filter_height, input_height - in_y_origin);
          float max = std::numeric_limits<float>::lowest();
          for (int filter_y = filter_y_start; filter_y < filter_y_end;
               ++filter_y) {
            for (int filter_x = filter_x_start; filter_x < filter_x_end;
                 ++filter_x) {
              const int in_x = in_x_origin + filter_x;
              const int in_y = in_y_origin + filter_y;
              max = std::max(
                  max,
                  input_data[Offset(input_shape, batch, in_y, in_x, channel)]);
            }
          }
          output_data[Offset(output_shape, batch, out_y, out_x, channel)] =
              ActivationFunctionWithMinMax(max, params.float_activation_min,
                                           params.float_activation_max);
        }
      }
    }
  }
}

inline void LocalResponseNormalization(
    const tflite::LocalResponseNormalizationParams& op_params,
    const RuntimeShape& input_shape, const float* input_data,
    const RuntimeShape& output_shape, float* output_data) {
  const int trailing_dim = input_shape.DimensionsCount() - 1;
  const int outer_size =
      MatchingFlatSizeSkipDim(input_shape, trailing_dim, output_shape);
  const int depth =
      MatchingDim(input_shape, trailing_dim, output_shape, trailing_dim);

  for (int i = 0; i < outer_size; ++i) {
    for (int c = 0; c < depth; ++c) {
      const int begin_input_c = std::max(0, c - op_params.range);
      const int end_input_c = std::min(depth, c + op_params.range);
      float accum = 0.f;
      for (int input_c = begin_input_c; input_c < end_input_c; ++input_c) {
        const float input_val = input_data[i * depth + input_c];
        accum += input_val * input_val;
      }
      const float multiplier =
          std::pow(op_params.bias + op_params.alpha * accum, -op_params.beta);
      output_data[i * depth + c] = input_data[i * depth + c] * multiplier;
    }
  }
}

template <typename SrcT, typename DstT>
inline void Cast(const RuntimeShape& input_shape, const SrcT* input_data,
                 const RuntimeShape& output_shape, DstT* output_data) {
  const int flat_size = MatchingFlatSize(input_shape, output_shape);

  for (int i = 0; i < flat_size; i++) {
    int offset = i;
    output_data[offset] = static_cast<DstT>(input_data[offset]);
  }
}

inline void Floor(const RuntimeShape& input_shape, const float* input_data,
                  const RuntimeShape& output_shape, float* output_data) {
  const int flat_size = MatchingFlatSize(input_shape, output_shape);

  for (int i = 0; i < flat_size; i++) {
    int offset = i;
    output_data[offset] = std::floor(input_data[offset]);
  }
}

template <typename T>
inline void Gather(const tflite::GatherParams& op_params,
                   const RuntimeShape& unextended_input_shape,
                   const T* input_data, const RuntimeShape& coords_shape,
                   const int32* coords_data,
                   const RuntimeShape& unextended_output_shape,
                   T* output_data) {
  TFLITE_DCHECK_LE(unextended_input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_LE(unextended_output_shape.DimensionsCount(), 4);
  const RuntimeShape input_shape =
      RuntimeShape::ExtendedShape(4, unextended_input_shape);
  const RuntimeShape output_shape =
      RuntimeShape::ExtendedShape(4, unextended_output_shape);

  const int input_rank = op_params.input_rank;
  const int gather_dimensions = output_shape.DimensionsCount();
  TFLITE_DCHECK_GE(input_shape.DimensionsCount(), gather_dimensions);
  const int axis = gather_dimensions - input_rank;
  TFLITE_DCHECK_LT(axis, gather_dimensions);
  TFLITE_DCHECK_GE(axis, 0);
  const int coords_count = coords_shape.FlatSize();
  TFLITE_DCHECK_EQ(coords_count, output_shape.Dims(axis));

  int64_t stride = 1;
  for (int i = axis + 1; i < gather_dimensions; ++i) {
    stride *= input_shape.Dims(i);
  }
  T* out = output_data;

  for (int i = 0; i < coords_count; ++i) {
    TFLITE_DCHECK_GE(coords_data[i], 0);
    TFLITE_DCHECK_LT(coords_data[i], input_shape.Dims(axis));
    const T* in = input_data + coords_data[i] * stride;
    memcpy(out, in, sizeof(T) * stride);
    out += stride;
  }
}

template <typename T>
inline void ResizeBilinear(const tflite::ResizeBilinearParams& op_params,
                           const RuntimeShape& unextended_input_shape,
                           const T* input_data,
                           const RuntimeShape& unextended_output_size_shape,
                           const int32* output_size_data,
                           const RuntimeShape& unextended_output_shape,
                           T* output_data) {
  TFLITE_DCHECK_LE(unextended_input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_LE(unextended_output_size_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_LE(unextended_output_shape.DimensionsCount(), 4);
  const RuntimeShape input_shape =
      RuntimeShape::ExtendedShape(4, unextended_input_shape);
  const RuntimeShape output_size_shape =
      RuntimeShape::ExtendedShape(4, unextended_output_size_shape);
  const RuntimeShape output_shape =
      RuntimeShape::ExtendedShape(4, unextended_output_shape);

  int32 batches = MatchingDim(input_shape, 0, output_shape, 0);
  int32 input_height = input_shape.Dims(1);
  int32 input_width = input_shape.Dims(2);
  int32 depth = MatchingDim(input_shape, 3, output_shape, 3);

  TFLITE_DCHECK_EQ(output_size_shape.Dims(0), 1);
  TFLITE_DCHECK_EQ(output_size_shape.Dims(1), 1);
  TFLITE_DCHECK_EQ(output_size_shape.Dims(2), 1);
  TFLITE_DCHECK_EQ(output_size_shape.Dims(3), 2);
  int32 output_height = output_size_data[Offset(output_size_shape, 0, 0, 0, 0)];
  int32 output_width = output_size_data[Offset(output_size_shape, 0, 0, 0, 1)];

  float height_scale = static_cast<float>(input_height) / output_height;
  float width_scale = static_cast<float>(input_width) / output_width;
  if (op_params.align_corners && output_height > 1) {
    height_scale = static_cast<float>(input_height - 1) / (output_height - 1);
  }
  if (op_params.align_corners && output_width > 1) {
    width_scale = static_cast<float>(input_width - 1) / (output_width - 1);
  }

  for (int b = 0; b < batches; ++b) {
    for (int y = 0; y < output_height; ++y) {
      float input_y = y * height_scale;
      int32 y0 = static_cast<int32>(std::floor(input_y));
      int32 y1 = std::min(y0 + 1, input_height - 1);
      for (int x = 0; x < output_width; ++x) {
        float input_x = x * width_scale;
        int32 x0 = static_cast<int32>(std::floor(input_x));
        int32 x1 = std::min(x0 + 1, input_width - 1);
        for (int c = 0; c < depth; ++c) {
          T interpolation =
              static_cast<T>(input_data[Offset(input_shape, b, y0, x0, c)] *
                                 (1 - (input_y - y0)) * (1 - (input_x - x0)) +
                             input_data[Offset(input_shape, b, y1, x0, c)] *
                                 (input_y - y0) * (1 - (input_x - x0)) +
                             input_data[Offset(input_shape, b, y0, x1, c)] *
                                 (1 - (input_y - y0)) * (input_x - x0) +
                             input_data[Offset(input_shape, b, y1, x1, c)] *
                                 (input_y - y0) * (input_x - x0));
          output_data[Offset(output_shape, b, y, x, c)] = interpolation;
        }
      }
    }
  }
}

template <typename T>
inline void SpaceToBatchND(
    const SpaceToBatchParams& params,
    const RuntimeShape& unextended_input1_shape, const T* input1_data,
    const RuntimeShape& unextended_input2_shape, const int32* block_shape_data,
    const RuntimeShape& unextended_input3_shape, const int32* paddings_data,
    const RuntimeShape& unextended_output_shape, T* output_data) {
  TFLITE_DCHECK_LE(unextended_input1_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_LE(unextended_output_shape.DimensionsCount(), 4);
  const RuntimeShape input1_shape =
      RuntimeShape::ExtendedShape(4, unextended_input1_shape);
  const RuntimeShape output_shape =
      RuntimeShape::ExtendedShape(4, unextended_output_shape);

  const int depth = input1_shape.Dims(3);
  const int input_width = input1_shape.Dims(2);
  const int input_height = input1_shape.Dims(1);
  const int input_batch_size = input1_shape.Dims(0);

  const int output_width = output_shape.Dims(2);
  const int output_height = output_shape.Dims(1);
  const int output_batch_size = output_shape.Dims(0);

  const int block_shape_height = block_shape_data[0];
  const int block_shape_width = block_shape_data[1];
  const int padding_top = paddings_data[0];
  const int padding_left = paddings_data[2];

  // For uint8 quantized, the correct padding "zero value" is the output offset.
  const int32_t pad_value = params.output_offset;

  for (int out_b = 0; out_b < output_batch_size; ++out_b) {
    int input_batch = out_b % input_batch_size;
    int shift_w = (out_b / input_batch_size) % block_shape_width;
    int shift_h = (out_b / input_batch_size) / block_shape_width;
    for (int out_h = 0; out_h < output_height; ++out_h) {
      for (int out_w = 0; out_w < output_width; ++out_w) {
        T* out = output_data + Offset(output_shape, out_b, out_h, out_w, 0);
        if (out_h * block_shape_height + shift_h < padding_top ||
            out_h * block_shape_height + shift_h >=
                padding_top + input_height ||
            out_w * block_shape_width + shift_w < padding_left ||
            out_w * block_shape_width + shift_w >= padding_left + input_width) {
          // This may not execute correctly when pad_value != 0 and T != uint8.
          memset(out, pad_value, depth * sizeof(T));
        } else {
          const T* in =
              input1_data +
              Offset(input1_shape, input_batch,
                     (out_h * block_shape_height + shift_h) - padding_top,
                     (out_w * block_shape_width + shift_w) - padding_left, 0);
          memcpy(out, in, depth * sizeof(T));
        }
      }
    }
  }
}

template <typename T>
inline void BatchToSpaceND(
    const RuntimeShape& unextended_input1_shape, const T* input1_data,
    const RuntimeShape& unextended_input2_shape, const int32* block_shape_data,
    const RuntimeShape& unextended_input3_shape, const int32* crops_data,
    const RuntimeShape& unextended_output_shape, T* output_data) {
  TFLITE_DCHECK_LE(unextended_input1_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_LE(unextended_output_shape.DimensionsCount(), 4);
  const RuntimeShape input1_shape =
      RuntimeShape::ExtendedShape(4, unextended_input1_shape);
  const RuntimeShape output_shape =
      RuntimeShape::ExtendedShape(4, unextended_output_shape);

  const int output_width = output_shape.Dims(2);
  const int output_height = output_shape.Dims(1);
  const int output_batch_size = output_shape.Dims(0);

  const int depth = input1_shape.Dims(3);
  const int input_width = input1_shape.Dims(2);
  const int input_height = input1_shape.Dims(1);
  const int input_batch_size = input1_shape.Dims(0);

  const int block_shape_width = block_shape_data[1];
  const int block_shape_height = block_shape_data[0];
  const int crops_top = crops_data[0];
  const int crops_left = crops_data[2];

  for (int in_batch = 0; in_batch < input_batch_size; ++in_batch) {
    const int out_batch = in_batch % output_batch_size;
    const int spatial_offset = in_batch / output_batch_size;
    for (int in_h = 0; in_h < input_height; ++in_h) {
      const int out_h = in_h * block_shape_height +
                        spatial_offset / block_shape_width - crops_top;
      if (out_h < 0 || out_h >= output_height) {
        continue;
      }
      for (int in_w = 0; in_w < input_width; ++in_w) {
        const int out_w = in_w * block_shape_width +
                          spatial_offset % block_shape_width - crops_left;

        if (out_w < 0 || out_w >= output_width) {
          continue;
        }
        T* out = output_data + Offset(output_shape, out_batch, out_h, out_w, 0);
        const T* in =
            input1_data + Offset(input1_shape, in_batch, in_h, in_w, 0);
        memcpy(out, in, depth * sizeof(T));
      }
    }
  }
}

// There are two versions of pad: Pad and PadV2.  In PadV2 there is a second
// scalar input that provides the padding value.  Therefore pad_value_ptr can be
// equivalent to a simple input1_data.  For Pad, it should point to a zero
// value.
//
// Note that two typenames are required, so that T=P=int32 is considered a
// specialization distinct from P=int32.
template <typename T, typename P>
inline void PadImpl(const tflite::PadParams& op_params,
                    const RuntimeShape& input_shape, const T* input_data,
                    const P* pad_value_ptr, const RuntimeShape& output_shape,
                    T* output_data) {
  const RuntimeShape ext_input_shape =
      RuntimeShape::ExtendedShape(4, input_shape);
  const RuntimeShape ext_output_shape =
      RuntimeShape::ExtendedShape(4, output_shape);
  TFLITE_DCHECK_LE(op_params.left_padding_count, 4);
  TFLITE_DCHECK_LE(op_params.right_padding_count, 4);

  // Runtime calls are currently fixed at 4 dimensions. Copy inputs so
  // we can pad them to 4 dims (yes, we are "padding the padding").
  std::vector<int> left_padding_copy(4, 0);
  for (int i = 0; i < op_params.left_padding_count; ++i) {
    left_padding_copy[i] = op_params.left_padding[i];
  }
  std::vector<int> right_padding_copy(4, 0);
  for (int i = 0; i < op_params.right_padding_count; ++i) {
    right_padding_copy[i] = op_params.right_padding[i];
  }

  const int output_batch = ext_output_shape.Dims(0);
  const int output_height = ext_output_shape.Dims(1);
  const int output_width = ext_output_shape.Dims(2);
  const int output_depth = ext_output_shape.Dims(3);

  const int left_b_padding = left_padding_copy[0];
  const int left_h_padding = left_padding_copy[1];
  const int left_w_padding = left_padding_copy[2];
  const int left_d_padding = left_padding_copy[3];

  const int right_b_padding = right_padding_copy[0];
  const int right_h_padding = right_padding_copy[1];
  const int right_w_padding = right_padding_copy[2];
  const int right_d_padding = right_padding_copy[3];

  const T pad_value = *pad_value_ptr;

  const T* in_ptr = input_data;
  T* out_ptr = output_data;
  for (int out_b = 0; out_b < output_batch; ++out_b) {
    for (int out_h = 0; out_h < output_height; ++out_h) {
      for (int out_w = 0; out_w < output_width; ++out_w) {
        for (int out_d = 0; out_d < output_depth; ++out_d) {
          if (out_b < left_b_padding ||
              out_b >= output_batch - right_b_padding ||
              out_h < left_h_padding ||
              out_h >= output_height - right_h_padding ||
              out_w < left_w_padding ||
              out_w >= output_width - right_w_padding ||
              out_d < left_d_padding ||
              out_d >= output_depth - right_d_padding) {
            *out_ptr++ = pad_value;
          } else {
            *out_ptr++ = *in_ptr++;
          }
        }
      }
    }
  }
}

template <typename T, typename P>
inline void Pad(const tflite::PadParams& op_params,
                const RuntimeShape& input_shape, const T* input_data,
                const P* pad_value_ptr, const RuntimeShape& output_shape,
                T* output_data) {
  PadImpl(op_params, input_shape, input_data, pad_value_ptr, output_shape,
          output_data);
}

// The second (pad-value) input can be int32 when, say, the first is uint8.
template <typename T>
inline void Pad(const tflite::PadParams& op_params,
                const RuntimeShape& input_shape, const T* input_data,
                const int32* pad_value_ptr, const RuntimeShape& output_shape,
                T* output_data) {
  const T converted_pad_value = static_cast<T>(*pad_value_ptr);
  PadImpl(op_params, input_shape, input_data, &converted_pad_value,
          output_shape, output_data);
}

// This version avoids conflicting template matching.
template <>
inline void Pad(const tflite::PadParams& op_params,
                const RuntimeShape& input_shape, const int32* input_data,
                const int32* pad_value_ptr, const RuntimeShape& output_shape,
                int32* output_data) {
  PadImpl(op_params, input_shape, input_data, pad_value_ptr, output_shape,
          output_data);
}

template <typename T>
inline void StridedSlice(const tflite::StridedSliceParams& op_params,
                         const RuntimeShape& unextended_input_shape,
                         const T* input_data,
                         const RuntimeShape& unextended_output_shape,
                         T* output_data) {
  // Note that the output_shape is not used herein.
  tflite::StridedSliceParams params_copy = op_params;

  TFLITE_DCHECK_LE(unextended_input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_LE(unextended_output_shape.DimensionsCount(), 4);
  const RuntimeShape input_shape =
      RuntimeShape::ExtendedShape(4, unextended_input_shape);
  const RuntimeShape output_shape =
      RuntimeShape::ExtendedShape(4, unextended_output_shape);

  // Reverse and pad to 4 dimensions because that is what the runtime code
  // requires (ie. all shapes must be 4D and are given backwards).
  strided_slice::StridedSlicePadIndices(&params_copy, 4);

  const int start_b = strided_slice::StartForAxis(params_copy, input_shape, 0);
  const int stop_b =
      strided_slice::StopForAxis(params_copy, input_shape, 0, start_b);
  const int start_h = strided_slice::StartForAxis(params_copy, input_shape, 1);
  const int stop_h =
      strided_slice::StopForAxis(params_copy, input_shape, 1, start_h);
  const int start_w = strided_slice::StartForAxis(params_copy, input_shape, 2);
  const int stop_w =
      strided_slice::StopForAxis(params_copy, input_shape, 2, start_w);
  const int start_d = strided_slice::StartForAxis(params_copy, input_shape, 3);
  const int stop_d =
      strided_slice::StopForAxis(params_copy, input_shape, 3, start_d);

  T* out_ptr = output_data;
  for (int in_b = start_b;
       !strided_slice::LoopCondition(in_b, stop_b, params_copy.strides[0]);
       in_b += params_copy.strides[0]) {
    for (int in_h = start_h;
         !strided_slice::LoopCondition(in_h, stop_h, params_copy.strides[1]);
         in_h += params_copy.strides[1]) {
      for (int in_w = start_w;
           !strided_slice::LoopCondition(in_w, stop_w, params_copy.strides[2]);
           in_w += params_copy.strides[2]) {
        for (int in_d = start_d; !strided_slice::LoopCondition(
                 in_d, stop_d, params_copy.strides[3]);
             in_d += params_copy.strides[3]) {
          *out_ptr++ = input_data[Offset(input_shape, in_b, in_h, in_w, in_d)];
        }
      }
    }
  }
}

template <typename T>
inline void Slice(const tflite::SliceParams& op_params,
                  const RuntimeShape& input_shape, const T* input_data,
                  const RuntimeShape& output_shape, T* output_data) {
  const RuntimeShape ext_shape = RuntimeShape::ExtendedShape(4, input_shape);
  // TODO(dkalenichenko): This op only supports 4D tensors or smaller.
  TFLITE_DCHECK_LE(op_params.begin_count, 4);
  TFLITE_DCHECK_LE(op_params.size_count, 4);
  const int begin_count = op_params.begin_count;
  const int size_count = op_params.size_count;
  // We front-pad the begin and size vectors.
  const int start_b = 4 - begin_count > 0 ? 0 : op_params.begin[0];
  const int stop_b = (4 - size_count > 0 || op_params.size[0] == -1)
                         ? ext_shape.Dims(0) - start_b
                         : start_b + op_params.size[0];
  const int start_h = begin_count < 3 ? 0 : op_params.begin[begin_count - 3];
  const int stop_h = (size_count < 3 || op_params.size[size_count - 3] == -1)
                         ? ext_shape.Dims(1) - start_h
                         : start_h + op_params.size[size_count - 3];
  const int start_w = begin_count < 2 ? 0 : op_params.begin[begin_count - 2];
  const int stop_w = (size_count < 2 || op_params.size[size_count - 2] == -1)
                         ? ext_shape.Dims(2) - start_w
                         : start_w + op_params.size[size_count - 2];
  const int start_d = begin_count < 1 ? 0 : op_params.begin[begin_count - 1];
  const int stop_d = (size_count < 1 || op_params.size[size_count - 1] == -1)
                         ? ext_shape.Dims(3) - start_d
                         : start_d + op_params.size[size_count - 1];

  T* out_ptr = output_data;
  for (int in_b = start_b; in_b < stop_b; ++in_b) {
    for (int in_h = start_h; in_h < stop_h; ++in_h) {
      for (int in_w = start_w; in_w < stop_w; ++in_w) {
        for (int in_d = start_d; in_d < stop_d; ++in_d) {
          *out_ptr++ = input_data[Offset(ext_shape, in_b, in_h, in_w, in_d)];
        }
      }
    }
  }
}

template <typename T>
inline void Exp(const T* input_data, const size_t num_elements,
                T* output_data) {
  for (size_t idx = 0; idx < num_elements; ++idx) {
    output_data[idx] = exp(input_data[idx]);
  }
}

// A generic reduce method that can be used for reduce_sum, reduce_mean, etc.
// This method iterates through input data and reduce elements along the
// dimensions given in axis.
template <typename In, typename Out>
inline bool Reduce(const In* input_data, const int* input_dims,
                   const int* output_dims, const int input_num_dims,
                   const int output_num_dims, const int* axis,
                   const int num_axis, int* input_iter,
                   Out reducer(const Out current, const In in),
                   Out* output_data) {
  // Reset input iterator.
  for (int idx = 0; idx < input_num_dims; ++idx) {
    input_iter[idx] = 0;
  }
  // Iterate through input_data.
  do {
    size_t input_offset =
        ReducedOutputOffset(input_num_dims, input_dims, input_iter, 0, nullptr);
    size_t output_offset = ReducedOutputOffset(input_num_dims, input_dims,
                                               input_iter, num_axis, axis);
    output_data[output_offset] =
        reducer(output_data[output_offset], input_data[input_offset]);
  } while (NextIndex(input_num_dims, input_dims, input_iter));
  return true;
}

inline bool ResolveAxis(const int num_dims, const int* axis,
                        const int64_t num_axis, int* out_axis,
                        int* out_num_axis) {
  *out_num_axis = 0;  // Just in case.
  // Short-circuit axis resolution for scalars; the axis will go unused.
  if (num_dims == 0) {
    return true;
  }
  // o(n^2) is fine since out_num_axis should be really small, mostly <= 4
  for (int64_t idx = 0; idx < num_axis; ++idx) {
    // Handle negative index.
    int current = axis[idx] < 0 ? (axis[idx] + num_dims) : axis[idx];
    TFLITE_DCHECK(current >= 0 && current < num_dims);
    bool is_dup = false;
    for (int j = 0; j < *out_num_axis; ++j) {
      if (out_axis[j] == current) {
        is_dup = true;
        break;
      }
    }
    if (!is_dup) {
      out_axis[*out_num_axis] = current;
      *out_num_axis += 1;
    }
  }
  return true;
}

// This method expects that output_data has been initialized.
template <typename In, typename Out>
inline bool ReduceSumImpl(const In* input_data, const int* input_dims,
                          const int* output_dims, const int input_num_dims,
                          const int output_num_dims, const int* axis,
                          const int num_axis, int* input_iter,
                          Out* output_data) {
  auto reducer = [](const Out current, const In in) -> Out {
    const Out actual_in = static_cast<Out>(in);
    return current + actual_in;
  };
  return Reduce<In, Out>(input_data, input_dims, output_dims, input_num_dims,
                         output_num_dims, axis, num_axis, input_iter, reducer,
                         output_data);
}

template <typename T>
inline bool InitTensorDataForReduce(const int* dims, const int num_dims,
                                    const T init_value, T* data) {
  size_t num_elements = 1;
  for (int idx = 0; idx < num_dims; ++idx) {
    size_t current = static_cast<size_t>(dims[idx]);
    // Overflow prevention.
    if (num_elements > std::numeric_limits<size_t>::max() / current) {
      return false;
    }
    num_elements *= current;
  }
  for (size_t idx = 0; idx < num_elements; ++idx) {
    data[idx] = init_value;
  }
  return true;
}

// Computes the generic value (i.e., sum/max/min/prod) of elements across
// dimensions given in axis. It needs to pass in init_value and reducer.
template <typename T>
inline bool ReduceGeneric(const T* input_data, const int* input_dims,
                          const int input_num_dims, T* output_data,
                          const int* output_dims, const int output_num_dims,
                          const int* axis, const int64_t num_axis_dimensions,
                          bool keep_dims, int* temp_index, int* resolved_axis,
                          T init_value,
                          T reducer(const T current, const T in)) {
  // Reset output data.
  if (!InitTensorDataForReduce(output_dims, output_num_dims, init_value,
                               output_data)) {
    return false;
  }

  // Resolve axis.
  int num_resolved_axis = 0;
  if (!ResolveAxis(input_num_dims, axis, num_axis_dimensions, resolved_axis,
                   &num_resolved_axis)) {
    return false;
  }

  return Reduce<T, T>(input_data, input_dims, output_dims, input_num_dims,
                      output_num_dims, resolved_axis, num_resolved_axis,
                      temp_index, reducer, output_data);
}

// Computes the mean of elements across dimensions given in axis.
// It does so in two stages, first calculates the sum of elements along the axis
// then divides it by the number of element in axis.
template <typename T, typename U>
inline bool Mean(const T* input_data, const int* input_dims,
                 const int input_num_dims, T* output_data,
                 const int* output_dims, const int output_num_dims,
                 const int* axis, const int num_axis_dimensions, bool keep_dims,
                 int* temp_index, int* resolved_axis, U* temp_sum) {
  // Reset output data.
  size_t num_outputs = 1;
  for (int idx = 0; idx < output_num_dims; ++idx) {
    size_t current = static_cast<size_t>(output_dims[idx]);
    // Overflow prevention.
    if (num_outputs > std::numeric_limits<size_t>::max() / current) {
      return false;
    }
    num_outputs *= current;
  }
  for (size_t idx = 0; idx < num_outputs; ++idx) {
    output_data[idx] = T();
    temp_sum[idx] = U();
  }

  // Resolve axis.
  int num_resolved_axis = 0;
  if (!ResolveAxis(input_num_dims, axis, num_axis_dimensions, resolved_axis,
                   &num_resolved_axis)) {
    return false;
  }

  if (!ReduceSumImpl<T, U>(input_data, input_dims, output_dims, input_num_dims,
                           output_num_dims, resolved_axis, num_resolved_axis,
                           temp_index, temp_sum)) {
    return false;
  }

  // Calculate mean by dividing output_data by num of aggregated element.
  U num_elements_in_axis = 1;
  for (int idx = 0; idx < num_resolved_axis; ++idx) {
    size_t current = static_cast<size_t>(input_dims[resolved_axis[idx]]);
    // Overflow prevention.
    if (current > (std::numeric_limits<U>::max() / num_elements_in_axis)) {
      return false;
    }
    num_elements_in_axis *= current;
  }

  if (num_elements_in_axis > 0) {
    for (size_t idx = 0; idx < num_outputs; ++idx) {
      output_data[idx] =
          static_cast<T>(temp_sum[idx] / static_cast<U>(num_elements_in_axis));
    }
  }
  return true;
}

template <typename T>
inline void Mean(const tflite::MeanParams& op_params,
                 const RuntimeShape& unextended_input_shape,
                 const T* input_data,
                 const RuntimeShape& unextended_output_shape, T* output_data) {
  TFLITE_DCHECK_LE(unextended_input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_LE(unextended_output_shape.DimensionsCount(), 4);
  const RuntimeShape input_shape =
      RuntimeShape::ExtendedShape(4, unextended_input_shape);
  const RuntimeShape output_shape =
      RuntimeShape::ExtendedShape(4, unextended_output_shape);

  const int output_batch = output_shape.Dims(0);
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);
  const int output_depth = output_shape.Dims(3);

  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);

  // The current implementation only supports simultaneous reduction over
  // width and height.
  TFLITE_DCHECK_EQ(op_params.axis_count, 2);
  TFLITE_DCHECK((op_params.axis[0] == 1 && op_params.axis[1] == 2) ||
                (op_params.axis[0] == 2 && op_params.axis[1] == 1));
  TFLITE_DCHECK_EQ(output_height, 1);
  TFLITE_DCHECK_EQ(output_width, 1);

  for (int out_b = 0; out_b < output_batch; ++out_b) {
    for (int out_d = 0; out_d < output_depth; ++out_d) {
      float value = 0;
      for (int in_h = 0; in_h < input_height; ++in_h) {
        for (int in_w = 0; in_w < input_width; ++in_w) {
          value += input_data[Offset(input_shape, out_b, in_h, in_w, out_d)];
        }
      }
      output_data[Offset(output_shape, out_b, 0, 0, out_d)] =
          value / (input_width * input_height);
    }
  }
}

template <typename T>
void Minimum(const RuntimeShape& input1_shape, const T* input1_data,
             const T* input2_data, const RuntimeShape& output_shape,
             T* output_data) {
  const int flat_size = MatchingFlatSize(input1_shape, output_shape);

  auto min_value = input2_data[0];
  for (int i = 0; i < flat_size; i++) {
    output_data[i] = input1_data[i] > min_value ? min_value : input1_data[i];
  }
}

// Convenience version that allows, for example, generated-code calls to be
// the same as other binary ops.
template <typename T>
inline void Minimum(const RuntimeShape& input1_shape, const T* input1_data,
                    const RuntimeShape&, const T* input2_data,
                    const RuntimeShape& output_shape, T* output_data) {
  // Drop shape of second input: not needed.
  Minimum(input1_shape, input1_data, input2_data, output_shape, output_data);
}

template <typename T>
void Maximum(const RuntimeShape& input1_shape, const T* input1_data,
             const T* input2_data, const RuntimeShape& output_shape,
             T* output_data) {
  const int flat_size = MatchingFlatSize(input1_shape, output_shape);

  auto max_value = input2_data[0];
  for (int i = 0; i < flat_size; i++) {
    output_data[i] = input1_data[i] < max_value ? max_value : input1_data[i];
  }
}

// Convenience version that allows, for example, generated-code calls to be
// the same as other binary ops.
template <typename T>
inline void Maximum(const RuntimeShape& input1_shape, const T* input1_data,
                    const RuntimeShape&, const T* input2_data,
                    const RuntimeShape& output_shape, T* output_data) {
  // Drop shape of second input: not needed.
  Maximum(input1_shape, input1_data, input2_data, output_shape, output_data);
}

template <typename T, typename Op>
void MaximumMinimumBroadcast4DSlow(const RuntimeShape& unextended_input1_shape,
                                   const T* input1_data,
                                   const RuntimeShape& unextended_input2_shape,
                                   const T* input2_data,
                                   const RuntimeShape& unextended_output_shape,
                                   T* output_data, Op op) {
  TFLITE_DCHECK_LE(unextended_input1_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_LE(unextended_input2_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_LE(unextended_output_shape.DimensionsCount(), 4);
  const RuntimeShape output_shape =
      RuntimeShape::ExtendedShape(4, unextended_output_shape);

  NdArrayDesc<4> desc1;
  NdArrayDesc<4> desc2;
  NdArrayDescsForElementwiseBroadcast(unextended_input1_shape,
                                      unextended_input2_shape, &desc1, &desc2);

  for (int b = 0; b < output_shape.Dims(0); ++b) {
    for (int y = 0; y < output_shape.Dims(1); ++y) {
      for (int x = 0; x < output_shape.Dims(2); ++x) {
        for (int c = 0; c < output_shape.Dims(3); ++c) {
          auto out_idx = Offset(output_shape, b, y, x, c);
          auto in1_idx = SubscriptToIndex(desc1, b, y, x, c);
          auto in2_idx = SubscriptToIndex(desc2, b, y, x, c);
          auto in1_val = input1_data[in1_idx];
          auto in2_val = input2_data[in2_idx];
          output_data[out_idx] = op(in1_val, in2_val);
        }
      }
    }
  }
}

template <typename T1, typename T2, typename T3, typename Cmp>
void ArgMinMax(const RuntimeShape& input1_shape, const T1* input1_data,
               const T3* input2_data, const RuntimeShape& output_shape,
               T2* output_data, const Cmp& cmp) {
  // The current ArgMax implemention can only determine the index of the maximum
  // value in the last dimension. So the axis argument is ignored.

  // For ArgMax, the number of output dimensions = (number of input dimensions -
  // 1). For the sake of simplicity, the output dimensions are equal to the
  // input dimensions here. We enforce the constraint that the last dimension
  // must always be 1.
  const int trailing_dim = output_shape.DimensionsCount() - 1;
  TFLITE_DCHECK_EQ(input1_shape.DimensionsCount(),
                   output_shape.DimensionsCount());
  TFLITE_DCHECK_EQ(output_shape.Dims(trailing_dim), 1);
  const int outer_size =
      MatchingFlatSizeSkipDim(input1_shape, trailing_dim, output_shape);
  const int depth = input1_shape.Dims(trailing_dim);

  for (int i = 0; i < outer_size; ++i) {
    auto min_max_value = input1_data[i * depth];
    int min_max_index = 0;
    for (int d = 1; d < depth; ++d) {
      const auto& curr_value = input1_data[i * depth + d];
      if (cmp(curr_value, min_max_value)) {
        min_max_value = curr_value;
        min_max_index = d;
      }
    }
    output_data[i] = min_max_index;
  }
}

template <typename T1, typename T2, typename T3>
void ArgMax(const RuntimeShape& input1_shape, const T1* input1_data,
            const T3* input2_data, const RuntimeShape& output_shape,
            T2* output_data) {
  ArgMinMax(input1_shape, input1_data, input2_data, output_shape, output_data,
            std::greater<T1>());
}

// Convenience version that allows, for example, generated-code calls to be
// the same as other binary ops.
template <typename T1, typename T2, typename T3>
inline void ArgMax(const RuntimeShape& input1_shape, const T1* input1_data,
                   const RuntimeShape& input2_shape, const T3* input2_data,
                   const RuntimeShape& output_shape, T2* output_data) {
  // Drop shape of second input: not needed.
  ArgMax(input1_shape, input1_data, input2_data, output_shape, output_data);
}

template <typename T>
void Transpose(const TransposeParams& params,
               const RuntimeShape& unextended_input_shape, const T* input_data,
               const RuntimeShape& unextended_output_shape, T* output_data) {
  const int unextended_output_size = unextended_output_shape.DimensionsCount();
  TFLITE_DCHECK_LE(unextended_input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_LE(unextended_output_size, 4);
  TFLITE_DCHECK_EQ(unextended_output_size, params.perm_count);
  const RuntimeShape input_shape =
      RuntimeShape::ExtendedShape(4, unextended_input_shape);
  const RuntimeShape output_shape =
      RuntimeShape::ExtendedShape(4, unextended_output_shape);
  const int input_ext_size = 4 - unextended_input_shape.DimensionsCount();
  const int output_ext_size = 4 - unextended_output_size;

  // The perm data is extended to match the output, each index incremented by
  // the amount of front padding of the input shape.
  int extended_perm[4];
  for (int i = 0; i < output_ext_size; ++i) {
    extended_perm[i] = i;
  }
  for (int i = 0; i < unextended_output_size; ++i) {
    extended_perm[i + output_ext_size] = params.perm[i] + input_ext_size;
  }

  int out_sizes[4];
  // Compute the inverse permutation array so we can do an output centered
  // transpose. Also, check to make sure output_dims is matching input_dims.
  for (int k = 0; k < 4; k++) {
    out_sizes[k] = MatchingDim(input_shape, extended_perm[k], output_shape, k);
  }

  // Naive transpose loop (iterate on output index and compute input index).
  int o[4];  // loop index (on output).
  int i[4];
  for (o[3] = 0; o[3] < out_sizes[3]; o[3]++) {
    i[extended_perm[3]] = o[3];
    for (o[2] = 0; o[2] < out_sizes[2]; o[2]++) {
      i[extended_perm[2]] = o[2];
      for (o[1] = 0; o[1] < out_sizes[1]; o[1]++) {
        i[extended_perm[1]] = o[1];
        for (o[0] = 0; o[0] < out_sizes[0]; o[0]++) {
          i[extended_perm[0]] = o[0];
          output_data[Offset(output_shape, o)] =
              input_data[Offset(input_shape, i)];
        }
      }
    }
  }
}

template <typename T>
inline bool EqualFn(T lhs, T rhs) {
  return lhs == rhs;
}

template <typename T>
inline bool NotEqualFn(T lhs, T rhs) {
  return lhs != rhs;
}

template <typename T>
inline bool GreaterFn(T lhs, T rhs) {
  return lhs > rhs;
}
template <typename T>
inline bool GreaterEqualFn(T lhs, T rhs) {
  return lhs >= rhs;
}
template <typename T>
inline bool LessFn(T lhs, T rhs) {
  return lhs < rhs;
}
template <typename T>
inline bool LessEqualFn(T lhs, T rhs) {
  return lhs <= rhs;
}

template <typename T>
using ComparisonFn = bool (*)(T, T);

template <typename T, ComparisonFn<T> F>
inline void ComparisonImpl(
    const ComparisonParams& op_params, const RuntimeShape& input1_shape,
    const T* input1_data, const RuntimeShape& input2_shape,
    const T* input2_data, const RuntimeShape& output_shape, bool* output_data) {
  const int64_t flatsize =
      MatchingFlatSize(input1_shape, input2_shape, output_shape);
  for (int64_t i = 0; i < flatsize; ++i) {
    output_data[i] = F(input1_data[i], input2_data[i]);
  }
}

template <ComparisonFn<float> F>
inline void Comparison(const ComparisonParams& op_params,
                       const RuntimeShape& input1_shape,
                       const float* input1_data,
                       const RuntimeShape& input2_shape,
                       const float* input2_data,
                       const RuntimeShape& output_shape, bool* output_data) {
  ComparisonImpl<float, F>(op_params, input1_shape, input1_data, input2_shape,
                           input2_data, output_shape, output_data);
}

template <typename T, ComparisonFn<T> F>
inline void BroadcastComparison4DSlowImpl(
    const ComparisonParams& op_params,
    const RuntimeShape& unextended_input1_shape, const T* input1_data,
    const RuntimeShape& unextended_input2_shape, const T* input2_data,
    const RuntimeShape& unextended_output_shape, bool* output_data) {
  TFLITE_DCHECK_LE(unextended_input1_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_LE(unextended_input2_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_LE(unextended_output_shape.DimensionsCount(), 4);
  const RuntimeShape output_shape =
      RuntimeShape::ExtendedShape(4, unextended_output_shape);

  NdArrayDesc<4> desc1;
  NdArrayDesc<4> desc2;
  NdArrayDescsForElementwiseBroadcast(unextended_input1_shape,
                                      unextended_input2_shape, &desc1, &desc2);

  for (int b = 0; b < output_shape.Dims(0); ++b) {
    for (int y = 0; y < output_shape.Dims(1); ++y) {
      for (int x = 0; x < output_shape.Dims(2); ++x) {
        for (int c = 0; c < output_shape.Dims(3); ++c) {
          output_data[Offset(output_shape, b, y, x, c)] =
              F(input1_data[SubscriptToIndex(desc1, b, y, x, c)],
                input2_data[SubscriptToIndex(desc2, b, y, x, c)]);
        }
      }
    }
  }
}
template <ComparisonFn<float> F>
inline void BroadcastComparison4DSlow(const ComparisonParams& op_params,
                                      const RuntimeShape& input1_shape,
                                      const float* input1_data,
                                      const RuntimeShape& input2_shape,
                                      const float* input2_data,
                                      const RuntimeShape& output_shape,
                                      bool* output_data) {
  BroadcastComparison4DSlowImpl<float, F>(op_params, input1_shape, input1_data,
                                          input2_shape, input2_data,
                                          output_shape, output_data);
}

#define TFLITE_COMPARISON_OP(name)                                             \
  inline void name(const ComparisonParams& op_params,                          \
                   const RuntimeShape& input1_shape, const float* input1_data, \
                   const RuntimeShape& input2_shape, const float* input2_data, \
                   const RuntimeShape& output_shape, bool* output_data) {      \
    Comparison<name##Fn>(op_params, input1_shape, input1_data, input2_shape,   \
                         input2_data, output_shape, output_data);              \
  }                                                                            \
  template <typename T>                                                        \
  inline void name##NoScaling(                                                 \
      const ComparisonParams& op_params, const RuntimeShape& input1_shape,     \
      const T* input1_data, const RuntimeShape& input2_shape,                  \
      const T* input2_data, const RuntimeShape& output_shape,                  \
      bool* output_data) {                                                     \
    ComparisonImpl<T, name##Fn>(op_params, input1_shape, input1_data,          \
                                input2_shape, input2_data, output_shape,       \
                                output_data);                                  \
  }                                                                            \
  template <typename T>                                                        \
  inline void Broadcast4DSlow##name##NoScaling(                                \
      const ComparisonParams& op_params, const RuntimeShape& input1_shape,     \
      const T* input1_data, const RuntimeShape& input2_shape,                  \
      const T* input2_data, const RuntimeShape& output_shape,                  \
      bool* output_data) {                                                     \
    BroadcastComparison4DSlowImpl<T, name##Fn>(                                \
        op_params, input1_shape, input1_data, input2_shape, input2_data,       \
        output_shape, output_data);                                            \
  }                                                                            \
  inline void Broadcast4DSlow##name(                                           \
      const ComparisonParams& op_params, const RuntimeShape& input1_shape,     \
      const float* input1_data, const RuntimeShape& input2_shape,              \
      const float* input2_data, const RuntimeShape& output_shape,              \
      bool* output_data) {                                                     \
    BroadcastComparison4DSlow<name##Fn>(op_params, input1_shape, input1_data,  \
                                        input2_shape, input2_data,             \
                                        output_shape, output_data);            \
  }
TFLITE_COMPARISON_OP(Equal);
TFLITE_COMPARISON_OP(NotEqual);
TFLITE_COMPARISON_OP(Greater);
TFLITE_COMPARISON_OP(GreaterEqual);
TFLITE_COMPARISON_OP(Less);
TFLITE_COMPARISON_OP(LessEqual);
#undef TFLITE_COMPARISON_OP

template <typename D, typename T>
void Select(const RuntimeShape& input_condition_shape,
            const D* input_condition_data, const RuntimeShape& input_x_shape,
            const T* input_x_data, const RuntimeShape& input_y_shape,
            const T* input_y_data, const RuntimeShape& output_shape,
            T* output_data) {
  const int64_t flatsize = MatchingFlatSize(
      input_condition_shape, input_x_shape, input_y_shape, output_shape);
  for (int64_t i = 0; i < flatsize; ++i) {
    output_data[i] =
        input_condition_data[i] ? input_x_data[i] : input_y_data[i];
  }
}

template <typename D, typename T>
void RankOneSelect(const RuntimeShape& input_condition_shape,
                   const D* input_condition_data,
                   const RuntimeShape& input_x_shape, const T* input_x_data,
                   const RuntimeShape& input_y_shape, const T* input_y_data,
                   const RuntimeShape& output_shape, T* output_data) {
  const int64_t outer_size = input_condition_shape.FlatSize();
  TFLITE_DCHECK_EQ(
      MatchingDim(input_x_shape, 0, input_y_shape, 0, output_shape, 0),
      outer_size);
  const int64_t inner_size =
      MatchingFlatSizeSkipDim(input_x_shape, 0, input_y_shape, output_shape);

  int64_t offset = 0;
  for (int64_t i = 0; i < outer_size; i++) {
    const T* input_data = input_condition_data[i] ? input_x_data : input_y_data;
    memcpy(output_data + offset, input_data + offset, inner_size * sizeof(T));
    offset += inner_size;
  }
}

// For easy implementation, the indices is always a vector of size-4 vectors.
template <typename T, typename TI>
inline void SparseToDense(const std::vector<std::vector<TI>>& indices,
                          const T* values, T default_value,
                          bool value_is_scalar,
                          const RuntimeShape& unextended_output_shape,
                          T* output_data) {
  TFLITE_DCHECK_LE(unextended_output_shape.DimensionsCount(), 4);
  const RuntimeShape output_shape =
      RuntimeShape::ExtendedShape(4, unextended_output_shape);
  const int value_count = indices.size();

  // First fill the output_data with default value.
  const int num_elements = output_shape.FlatSize();
  for (int i = 0; i < num_elements; ++i) {
    output_data[i] = default_value;
  }

  // Special handle for value is scalar case to avoid checking the boolean
  // condition within the loop every time.
  if (value_is_scalar) {
    for (int i = 0; i < value_count; ++i) {
      const std::vector<TI>& index = indices[i];
      TFLITE_DCHECK_EQ(index.size(), 4);
      const T value = *values;  // just use the first value.
      output_data[Offset(output_shape, index[0], index[1], index[2],
                         index[3])] = value;
    }
    return;
  }

  // Go through the values and indices to fill the sparse values.
  for (int i = 0; i < value_count; ++i) {
    const std::vector<TI>& index = indices[i];
    TFLITE_DCHECK_EQ(index.size(), 4);
    const T value = values[i];
    output_data[Offset(output_shape, index[0], index[1], index[2], index[3])] =
        value;
  }
}

template <typename T>
inline void Pow(const RuntimeShape& input1_shape, const T* input1_data,
                const RuntimeShape& input2_shape, const T* input2_data,
                const RuntimeShape& output_shape, T* output_data) {
  const int flat_size =
      MatchingFlatSize(input1_shape, input2_shape, output_shape);
  for (int i = 0; i < flat_size; ++i) {
    output_data[i] = std::pow(input1_data[i], input2_data[i]);
  }
}

template <typename T>
inline void BroadcastPow4DSlow(const RuntimeShape& unextended_input1_shape,
                               const T* input1_data,
                               const RuntimeShape& unextended_input2_shape,
                               const T* input2_data,
                               const RuntimeShape& unextended_output_shape,
                               T* output_data) {
  TFLITE_DCHECK_LE(unextended_input1_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_LE(unextended_input2_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_LE(unextended_output_shape.DimensionsCount(), 4);
  const RuntimeShape output_shape =
      RuntimeShape::ExtendedShape(4, unextended_output_shape);

  NdArrayDesc<4> desc1;
  NdArrayDesc<4> desc2;
  NdArrayDescsForElementwiseBroadcast(unextended_input1_shape,
                                      unextended_input2_shape, &desc1, &desc2);

  for (int b = 0; b < output_shape.Dims(0); ++b) {
    for (int y = 0; y < output_shape.Dims(1); ++y) {
      for (int x = 0; x < output_shape.Dims(2); ++x) {
        for (int c = 0; c < output_shape.Dims(3); ++c) {
          auto out_idx = Offset(output_shape, b, y, x, c);
          auto in1_idx = SubscriptToIndex(desc1, b, y, x, c);
          auto in2_idx = SubscriptToIndex(desc2, b, y, x, c);
          auto in1_val = input1_data[in1_idx];
          auto in2_val = input2_data[in2_idx];
          output_data[out_idx] = std::pow(in1_val, in2_val);
        }
      }
    }
  }
}

inline void Logical(const RuntimeShape& input1_shape, const bool* input1_data,
                    const RuntimeShape& input2_shape, const bool* input2_data,
                    const RuntimeShape& output_shape, bool* output_data,
                    const std::function<bool(bool, bool)>& func) {
  const int flat_size =
      MatchingFlatSize(input1_shape, input2_shape, output_shape);
  for (int i = 0; i < flat_size; ++i) {
    output_data[i] = func(input1_data[i], input2_data[i]);
  }
}

inline void BroadcastLogical4DSlow(
    const RuntimeShape& unextended_input1_shape, const bool* input1_data,
    const RuntimeShape& unextended_input2_shape, const bool* input2_data,
    const RuntimeShape& unextended_output_shape, bool* output_data,
    const std::function<bool(bool, bool)>& func) {
  TFLITE_DCHECK_LE(unextended_input1_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_LE(unextended_input2_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_LE(unextended_output_shape.DimensionsCount(), 4);
  const RuntimeShape output_shape =
      RuntimeShape::ExtendedShape(4, unextended_output_shape);

  NdArrayDesc<4> desc1;
  NdArrayDesc<4> desc2;
  NdArrayDescsForElementwiseBroadcast(unextended_input1_shape,
                                      unextended_input2_shape, &desc1, &desc2);

  for (int b = 0; b < output_shape.Dims(0); ++b) {
    for (int y = 0; y < output_shape.Dims(1); ++y) {
      for (int x = 0; x < output_shape.Dims(2); ++x) {
        for (int c = 0; c < output_shape.Dims(3); ++c) {
          auto out_idx = Offset(output_shape, b, y, x, c);
          auto in1_idx = SubscriptToIndex(desc1, b, y, x, c);
          auto in2_idx = SubscriptToIndex(desc2, b, y, x, c);
          auto in1_val = input1_data[in1_idx];
          auto in2_val = input2_data[in2_idx];
          output_data[out_idx] = func(in1_val, in2_val);
        }
      }
    }
  }
}

// TODO(ycling): Refactoring. Remove BroadcastLogical and use the more
// generalized and efficient BroadcastBinaryFunction.
//
// Also appears to duplicte MinimumMaximum.
//
// R: Result type. T1: Input 1 type. T2: Input 2 type.
template <typename R, typename T1, typename T2>
inline void BroadcastBinaryFunction4DSlow(
    const RuntimeShape& unextended_input1_shape, const T1* input1_data,
    const RuntimeShape& unextended_input2_shape, const T2* input2_data,
    const RuntimeShape& unextended_output_shape, R* output_data,
    R (*func)(T1, T2)) {
  TFLITE_DCHECK_LE(unextended_input1_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_LE(unextended_input2_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_LE(unextended_output_shape.DimensionsCount(), 4);
  const RuntimeShape output_shape =
      RuntimeShape::ExtendedShape(4, unextended_output_shape);

  NdArrayDesc<4> desc1;
  NdArrayDesc<4> desc2;
  NdArrayDescsForElementwiseBroadcast(unextended_input1_shape,
                                      unextended_input2_shape, &desc1, &desc2);

  for (int b = 0; b < output_shape.Dims(0); ++b) {
    for (int y = 0; y < output_shape.Dims(1); ++y) {
      for (int x = 0; x < output_shape.Dims(2); ++x) {
        for (int c = 0; c < output_shape.Dims(3); ++c) {
          auto out_idx = Offset(output_shape, b, y, x, c);
          auto in1_idx = SubscriptToIndex(desc1, b, y, x, c);
          auto in2_idx = SubscriptToIndex(desc2, b, y, x, c);
          auto in1_val = input1_data[in1_idx];
          auto in2_val = input2_data[in2_idx];
          output_data[out_idx] = func(in1_val, in2_val);
        }
      }
    }
  }
}

// R: Result type. T1: Input 1 type. T2: Input 2 type.
// TODO(renjieliu): Refactor other binary functions to use this one.
template <typename R, typename T1, typename T2>
inline void BinaryFunction(const RuntimeShape& input1_shape,
                           const T1* input1_data,
                           const RuntimeShape& input2_shape,
                           const T2* input2_data,
                           const RuntimeShape& output_shape, R* output_data,
                           R (*func)(T1, T2)) {
  const int flat_size =
      MatchingFlatSize(input1_shape, input2_shape, output_shape);
  for (int i = 0; i < flat_size; ++i) {
    output_data[i] = func(input1_data[i], input2_data[i]);
  }
}

}  // namespace reference_ops
}  // namespace tflite

#endif  // TENSORFLOW_CONTRIB_LITE_EXPERIMENTAL_RISCV_KERNELS_REFERENCE_REFERENCE_OPS_FLOAT_H_
