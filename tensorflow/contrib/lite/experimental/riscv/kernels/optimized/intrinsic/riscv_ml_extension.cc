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

#ifdef RISCV

void AddBiasActivationFunctionWithMinMax(float* vector, const float* bias,
                                         float output_min, float output_max,
                                         int flatted_len, int bias_len){

  int new_len = bias_len - (bias_len & (kMaxVectorLength32 - 1));
  int len_diff = bias_len & (kMaxVectorLength32 - 1);

  __VectorBroadcastMinMax(output_min, output_max);

  for (int fl = 0; fl < flatted_len; fl+= bias_len) {
    for (int i = 0; i < new_len; i+= kMaxVectorLength32) {
      __VectorLoad((vector + i + fl), (bias + i));
      __VectorAddFloat();
      __VectorMinMaxFloat();
      __VectorStore((vector + i + fl));
    }

    if (len_diff != 0) {
      SetVl(len_diff);
      __VectorLoad((vector + new_len  + fl), (bias + new_len));
      __VectorAddFloat();
      __VectorMinMaxFloat();
      __VectorStore(vector + new_len + fl);

    }
  }
}

void VectorVectorAdd(const float* input1, const float* input2, float* output,
                     int len) {
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
                           int len) {
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
                          float* output, int len) {
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
                                    float* output, int len) {
  int new_len = len - (len & (kMaxVectorLength32 - 1));
  int len_diff = len & (kMaxVectorLength32 - 1);

   SetVl(kMaxVectorLength32);
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

void VectorBatchVectorCwiseProductAccumulate(const float* vector, int v_len,
                                             const float* batch_vector,
                                             int batch_size,
                                             float* output) {
  int new_len = v_len - (v_len & (kMaxVectorLength32 - 1));
  int len_diff = v_len & (kMaxVectorLength32 - 1);
  SetConfig(kElementWidthMax32, kMaxVectorLength32);

  float* output_ptr = output;
  const float* batch_vector_ptr = batch_vector;
  for (int b =0; b < batch_size; b++){
    for (int i = 0; i < new_len; i += kMaxVectorLength32) {
      __VectorLoadPartialOutput(output_ptr + i);
      __VectorLoad((vector + i), (batch_vector_ptr + i));
      __VectorMulAccFloat();
    }
    if (len_diff != 0) {
      SetVl(len_diff);
      __VectorLoad((vector + new_len), (batch_vector_ptr + new_len));
      SetVl(kMaxVectorLength32);
      __VectorMulAccFloat();
    }

    SetVl(kMaxVectorLength32);
    __VectorReduceAccumFloat();  // Reduce partial sum to single value
    __VectorStoreAccum(output_ptr);  // Store as scalar value

    output_ptr += v_len;
    batch_vector_ptr += v_len;
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

void MatrixVectorMultiplyAccumulate(const float* matrix,
                                    int matrix_rows,
                                    int matrix_cols,
                                    const float* vector,
                                    float* output) {
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

void MatrixBatchVectorMultiplyAccumulate(const float* matrix,
                                         int matrix_rows,
                                         int matrix_cols,
                                         const float* vector,
                                         int batch_size,
                                         float* output,
                                         int output_stride) {
  // Vector length is equal to # columns
  int new_cols = matrix_cols - (matrix_cols & (kMaxVectorLength32 - 1));
  int col_diff = matrix_cols & (kMaxVectorLength32 - 1);

  SetConfig(kElementWidthMax32, kMaxVectorLength32);

  for (int b =0; b < batch_size; b++) {
    float* output_batch = output + b * matrix_rows * output_stride;
    const float* vector_batch = vector + b * matrix_cols;
    const float* matrix_row = matrix;

    for (int r = 0; r < matrix_rows; r++) {
      __VectorBroadcastAccum(0.0);
      for (int c = 0; c < new_cols; c += kMaxVectorLength32) {
        __VectorLoad(vector_batch + c, matrix_row + c);
        __VectorMulAccFloat();
      }

      if (col_diff != 0) {
        SetVl(col_diff);
        __VectorLoad((vector + new_cols), (matrix_row + new_cols));
        SetVl(kMaxVectorLength32);
        __VectorMulAccFloat();
      }
      SetVl(kMaxVectorLength32);
      __VectorReduceAccumFloat();      // Reduce partial sum to single value
      __VectorStoreAccum(output_batch);  // Store as scalar value
      matrix_row += matrix_cols;
      output_batch += output_stride;
    }
  }
}

// void Kernel1x1MultiplyAccumulate(const float* filter,
//                                  int input_depth,
//                                  int output_depth,
//                                  const float* input,
//                                  float* output) {

//   // special matrix multiplication for 1 x 1 kernel
//   SetConfig(kElementWidthMax32, kMaxVectorLength32);

//   int new_output_depth = output_depth - (output_depth & (kMaxVectorLength32 - 1));
//   int output_depth_diff = output_depth & (kMaxVectorLength32 - 1);

//   int new_input_depth = input_depth - (input_depth & (kMaxVectorLength32 - 1));
//   int input_depth_diff = input_depth & (kMaxVectorLength32 - 1);

//   for(int zout = 0; zout < new_output_depth; zout += kMaxVectorLength32){
//     __VectorBroadcastAccum(0.0);
//     const float* filter_ptr = filter + input_depth*zout;
//     for(int zin = 0; zin < new_input_depth; zin += kMaxVectorLength32){
//       __VectorLoadInput1(input + zin);
//       __VectorLoadKernel11kMaxVectorLength32(filter_ptr + zin);
//       __MatrixVectorMulAccumulatekMaxVectorLength32()
//     }
//     if (input_depth_diff){
//       SetVl(input_depth_diff);
//       __VectorLoadInput1(input + new_input_depth);
//       __VectorLoadKernel11(filter_ptr + new_input_depth,
//                          input_depth_diff);
//       SetVl(kMaxVectorLength32);
//         __MatrixVectorMulAccumulate(input_depth_diff);
//     }
//     __VectorStore(output+zout);
//   }
//   if(output_depth_diff){
//     for(int zout = new_output_depth; zout < output_depth; zout ++){
//       __VectorBroadcastAccum(0.0);
//       const float* filter_ptr = filter + input_depth * zout;
//       for(int zin = 0; zin < new_input_depth; zin += kMaxVectorLength32){
//         __VectorLoad((input + zin), (filter_ptr + zin));
//         __VectorMulAccFloat();
//       }
//       if (input_depth_diff) {
//         SetVl(input_depth_diff);
//         __VectorLoad((input + new_input_depth), (filter_ptr + new_input_depth));
//         SetVl(kMaxVectorLength32);
//         __VectorMulAccFloat();
//       }
//       __VectorReduceAccumFloat();  // Reduce partial sum to single value
//       __VectorStoreAccum(output + zout);  // Store as scalar value
//     }
//   }
// }

void Kernel1x1MultiplyAccumulate(const float* filter,
                                 int input_depth,
                                 int output_depth,
                                 const float* input,
                                 float* output) {

  // special matrix multiplication for 1 x 1 kernel
  SetConfig(kElementWidthMax32, kMaxVectorLength32);

  int new_output_depth = output_depth - (output_depth & (kMaxVectorLength32 - 1));
  int output_depth_diff = output_depth & (kMaxVectorLength32 - 1);

  int new_input_depth = input_depth - (input_depth & (kMaxVectorLength32 - 1));
  int input_depth_diff = input_depth & (kMaxVectorLength32 - 1);

  for(int zout = 0; zout < new_output_depth; zout += kMaxVectorLength32){
    __VectorBroadcastAccum(0.0);
    const float* filter_ptr = filter + input_depth*zout;
    for(int zin = 0; zin < new_input_depth; zin += kMaxVectorLength32){
      __VectorLoadInput1(input + zin);
      __VectorLoadInput2(filter_ptr + zin, input_depth);

      // TODO(marcialouis) : We need to write it as marcos
      for(int i =0 ; i< kMaxVectorLength32; i++) {
        __VectorSplatMulAccFloat(i);
      }
    }
    if (input_depth_diff){
      SetVl(input_depth_diff);
      __VectorLoadInput1(input + new_input_depth);
      __VectorLoadInput2(filter_ptr + new_input_depth,
                         input_depth);
      SetVl(kMaxVectorLength32);
      for(int i =0 ; i< input_depth_diff; i++) {
        __VectorSplatMulAccFloat(i);
      }
    }
    __VectorStore(output+zout);
  }
  if(output_depth_diff){
    for(int zout = new_output_depth; zout < output_depth; zout ++){
      __VectorBroadcastAccum(0.0);
      const float* filter_ptr = filter + input_depth * zout;
      for(int zin = 0; zin < new_input_depth; zin += kMaxVectorLength32){
        __VectorLoad((input + zin), (filter_ptr + zin));
        __VectorMulAccFloat();
      }
      if (input_depth_diff) {
        SetVl(input_depth_diff);
        __VectorLoad((input + new_input_depth), (filter_ptr + new_input_depth));
        SetVl(kMaxVectorLength32);
        __VectorMulAccFloat();
      }
      __VectorReduceAccumFloat();  // Reduce partial sum to single value
      __VectorStoreAccum(output + zout);  // Store as scalar value
    }
  }
}

#endif
