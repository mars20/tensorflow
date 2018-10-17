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
#include "tensorflow/contrib/lite/experimental/riscv/kernels/optimized/intrinsic/riscv_ml_extension.h"
#include <malloc.h>

void VectorVectorAdd(const float* input1, const float* input2, float* output,
                     int len, int batch_size) {
  int new_len = len - (len & (kMaxVectorLength32 - 1));
  int len_diff = len & (kMaxVectorLength32 - 1);

  SetConfig(kElementWidthMax32, kMaxVectorLength32);

  for (int i = 0; i < new_len; i += kMaxVectorLength32) {
    __VectorLoad((input1 + i), (input2 + i));
    __VectorAddFloat();
    __VectorStore((output + i));
  }

  if (len_diff != 0) {
    SetVl(len_diff);
    __VectorLoad((input1 + new_len), (input2 + new_len));
    __VectorAddFloat();
    __VectorStore((output + new_len));
  }
}

void VectorVectorAddMinMax(const float* input1, const float* input2,
                           float output_min, float output_max, float* output,
                           int len, int batch_size) {
  int new_len = len - (len & (kMaxVectorLength32 - 1));
  int len_diff = len & (kMaxVectorLength32 - 1);

  SetConfig(kElementWidthMax32, kMaxVectorLength32);

  __VectorBroadcastMinMax(output_min, output_max);

  for (int i = 0; i < new_len; i += kMaxVectorLength32) {
    // TODO: check if BroadcastMinMax is called first
    __VectorLoad((input1 + i), (input2 + i));
    __VectorAddFloat();
    __VectorMinMaxFloat();
    __VectorStore((output + i));
  }

  if (len_diff != 0) {
    // TODO: check if BroadcastMinMax is called first
    SetVl(len_diff);
    __VectorLoad((input1 + new_len), (input2 + new_len));
    __VectorAddFloat();
    __VectorMinMaxFloat();
    __VectorStore((output + new_len));
  }
}

void VectorVectorMultiply(const float* input1, const float* input2,
                          float* output, int len, int batch_size) {
  int new_len = len - (len & (kMaxVectorLength32 - 1));
  int len_diff = len & (kMaxVectorLength32 - 1);

  SetConfig(kElementWidthMax32, kMaxVectorLength32);

  for (int i = 0; i < new_len; i += kMaxVectorLength32) {
    __VectorLoad((input1 + i), (input2 + i));
    __VectorMulFloat();
    __VectorStore((output + i));
  }

  if (len_diff != 0) {
    SetVl(len_diff);
    __VectorLoad((input1 + new_len), (input2 + new_len));
    __VectorMulFloat();
    __VectorStore((output + new_len));
  }
}

void VectorVectorMultiplyAccumulate(const float* input1, const float* input2,
                                    float* output, int len, int batch_size) {
  int new_len = len - (len & (kMaxVectorLength32 - 1));
  int len_diff = len & (kMaxVectorLength32 - 1);
  SetConfig(kElementWidthMax32, kMaxVectorLength32);

  __VectorBroadcastAccum(0.0);
  for (int i = 0; i < new_len; i += kMaxVectorLength32) {
    __VectorLoad((input1 + i), (input2 + i));
    __VectorMulAccFloat();
  }
  if (len_diff != 0) {
    SetVl(len_diff);
    __VectorLoad((input1 + new_len), (input2 + new_len));
    SetVl(kMaxVectorLength32);
    __VectorMulAccFloat();
  }

  SetVl(kMaxVectorLength32);
  __VectorReduceAccumFloat();  // Reduce partial sum to single value
  __VectorStoreAccum(output);  // Store as scalar value
}

void VectorMatrixMultiplyAccumulate(const float* vector, const float* matrix,
                                    float* output, int matrix_rows,
                                    int matrix_cols, int batch_size) {
  // Vector length is equal to # columns
  // Output length is equal to # rows

  int new_cols = matrix_cols - (matrix_cols & (kMaxVectorLength32 - 1));
  int col_diff = matrix_cols & (kMaxVectorLength32 - 1);

  SetConfig(kElementWidthMax32, kMaxVectorLength32);

  for (int r = 0; r < matrix_rows; r++) {
    __VectorBroadcastAccum(0.0);
    for (int c = 0; c < new_cols; c += kMaxVectorLength32) {
      __VectorLoad(vector + c, matrix + r * matrix_cols + c);
      __VectorMulAccFloat();
    }

    if (col_diff != 0) {
      SetVl(col_diff);
      __VectorLoad((vector + new_cols), (matrix + r * matrix_cols + new_cols));
      SetVl(kMaxVectorLength32);
      __VectorMulAccFloat();
    }
    SetVl(kMaxVectorLength32);
    __VectorReduceAccumFloat();      // Reduce partial sum to single value
    __VectorStoreAccum(output + r);  // Store as scalar value
  }
}

void VectorVectorMultiplyAccumulateDepthwise(const float* input1,
                                             const float* input2, float* output,
                                             int depth, int depth_multiplier,
                                             const float* bias, bool use_bias) {
  int new_depth = depth - (depth & (kMaxVectorLength32 - 1));
  int depth_diff = depth & (kMaxVectorLength32 - 1);
  SetConfig(kElementWidthMax32, kMaxVectorLength32);



  for (int i = 0; i < new_depth; i += kMaxVectorLength32) {
    __VectorLoadInput1((input1 + i));
    for (int m = 0; m < depth_multiplier; m++) {
      if (use_bias) {
        __VectorLoadBias(bias + i + m, depth_multiplier);
      } else {
        __VectorLoadPartialOutput(output + i + m, depth_multiplier);
      }
      __VectorLoadInput2((input2 + (i + m)), depth_multiplier);
      __VectorMulAccFloat();
      __VectorStorePartialOutput(output + i + m, depth_multiplier);
    }
  }
  if (depth_diff != 0) {
    SetVl(depth_diff);
    __VectorLoadInput1((input1 + new_depth));
    for (int m = 0; m < depth_multiplier; m++) {
      if (use_bias) {
        __VectorLoadBias(bias + new_depth + m, depth_multiplier);
      } else {
        __VectorLoadPartialOutput(output + new_depth + m, depth_multiplier);
      }
      __VectorLoadInput2((input2 + (new_depth + m)), depth_multiplier);
      __VectorMulAccFloat();
      __VectorStorePartialOutput(output + new_depth + m, depth_multiplier);
    }
  }
}

void VectorActivationFunctionWithMinMax(float* data, float activation_min, float activation_max, int length){
  int new_length = length - (length & (kMaxVectorLength32 - 1));
  int length_diff = length & (kMaxVectorLength32 - 1);

  SetConfig(kElementWidthMax32, kMaxVectorLength32);

  __VectorBroadcastMinMax(activation_min, activation_max);

  for (int i = 0; i < new_length; i += kMaxVectorLength32) {
    __VectorLoadActivationInput(data + i);
    __VectorMinMaxFloat();
    __VectorStore((data + i));
  }

  if (length_diff != 0) {
    // TODO: check if BroadcastMinMax is called first
    SetVl(length_diff);
    __VectorLoadActivationInput(data + new_length);
    __VectorMinMaxFloat();
    __VectorStore((data + new_length));
  }
}

void VectorAveragePooling(const float* input, float* output, int depth,
                          bool use_zero) {
  int new_depth = depth - (depth & (kMaxVectorLength32 - 1));
  int depth_diff = depth & (kMaxVectorLength32 - 1);
  SetConfig(kElementWidthMax32, kMaxVectorLength32);

  for (int i = 0; i < new_depth; i += kMaxVectorLength32) {
    __VectorLoadInput1((input + i));
    if (use_zero) {
      __VectorBroadcastAccum(0.0);
    } else {
      __VectorLoadPartialOutput(output + i);
    }
    __VectorAccFloat();
    __VectorStorePartialOutput(output + i);
  }
  if (depth_diff != 0) {
    SetVl(depth_diff);
    __VectorLoadInput1((input + new_depth));
    if (use_zero) {
      __VectorBroadcastAccum(0.0);
    } else {
      __VectorLoadPartialOutput(output + new_depth);
    }
    __VectorAccFloat();
    __VectorStorePartialOutput(output + new_depth);
  }
}
