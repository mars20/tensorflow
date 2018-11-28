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
#include <cstdio>
#include <cstdlib>

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

      // TODO(marcialouis) : We need to write it as marcos
       __VectorLoadInput2(filter_ptr, input_depth);
       __VectorSplatMulAccFloat(0);
       __VectorLoadInput2(filter_ptr + 1, input_depth);
       __VectorSplatMulAccFloat(1);
       __VectorLoadInput2(filter_ptr + 2, input_depth);
       __VectorSplatMulAccFloat(2);
       __VectorLoadInput2(filter_ptr + 3, input_depth);
       __VectorSplatMulAccFloat(3);
       // __VectorLoadInput2(filter_ptr + 4, input_depth);
       // __VectorSplatMulAccFloat(4);
       // __VectorLoadInput2(filter_ptr + 5, input_depth);
       // __VectorSplatMulAccFloat(5);
       // __VectorLoadInput2(filter_ptr + 6, input_depth);
       // __VectorSplatMulAccFloat(6);
       // __VectorLoadInput2(filter_ptr + 7, input_depth);
       // __VectorSplatMulAccFloat(7);
      // for(int i =0 ; i< kMaxVectorLength32; i++) {
      //   __VectorLoadInput2(filter_ptr + i, input_depth);
      //   __VectorSplatMulAccFloat(i);
      // }
    }
    if (input_depth_diff){
      SetVl(input_depth_diff);
      __VectorLoadInput1(input + new_input_depth);

      SetVl(kMaxVectorLength32);
      for(int i =0 ; i< input_depth_diff; i++) {
        __VectorLoadInput2(filter_ptr + i,
                         input_depth);
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

void MatrixMatrixMultiplyAccumulate4x1(const float* matrix_a,
                                       int matrix_a_rows,
                                       int matrix_a_cols,
                                       const float* matrix_b,
                                       int matrix_b_rows,
                                       int matrix_b_cols,
                                       float* matrix_c) {

  // Matrix A (input) is layed out memory as BHWC
  // Matrix B (filter) is layed out in memory as FHWC
  // Matrix C (output) is layed out in memory as BHWF
  if(matrix_a_cols != matrix_b_cols) {
    fprintf(stderr, "Input and filter depth should match %d != %d"
            , matrix_a_cols, matrix_b_cols);
    exit(1);
  }

  // printf("Matrix A %dx%d \n",matrix_a_rows, matrix_a_cols);
  // printf("Matrix B %dx%d \n",matrix_b_rows, matrix_b_cols);

  // for(int i =0; i<matrix_a_rows; i++){
  //   for(int j=0; j< matrix_a_cols; j++){
  //     printf("%f\t", matrix_a[j+i*matrix_a_cols]);
  //   }
  //   printf("\n");
  // }
  // We know the matrix_b_rows <= kMaxVectorLength32
  //  int block_size = 1;

  SetConfig(kElementWidthMax32, matrix_b_rows);

  int new_matrix_a_rows =  matrix_a_rows - (matrix_a_rows & (MAX_BLOCK_SIZE - 1));
  int matrix_a_diff = matrix_a_rows & (MAX_BLOCK_SIZE - 1);


  for(int row_idx_a = 0; row_idx_a < new_matrix_a_rows; row_idx_a += MAX_BLOCK_SIZE) {
    // clear the accumulator registers
    __VectorClearOutputAccumBlock4x1();

    // Compute base address of # MAX_BLOCK_SIZE addresses in matrix_a
    const float* matrix_a_ptr_block1 = matrix_a + matrix_a_cols * row_idx_a;
    const float* matrix_a_ptr_block2 = matrix_a + matrix_a_cols * (row_idx_a+1);
    const float* matrix_a_ptr_block3 = matrix_a + matrix_a_cols * (row_idx_a+2);
    const float* matrix_a_ptr_block4 = matrix_a + matrix_a_cols * (row_idx_a+3);

    for(int col_idx_a = 0; col_idx_a < matrix_a_cols; col_idx_a++){

      __VectorLoadInput1Block4(matrix_a_ptr_block1 + col_idx_a,
                               matrix_a_ptr_block2 + col_idx_a,
                               matrix_a_ptr_block3 + col_idx_a,
                               matrix_a_ptr_block4 + col_idx_a);

      // printf("Input: %f, %f, %f, %f\n", *(matrix_a_ptr_block1 + col_idx_a),
      //        *(matrix_a_ptr_block2 + col_idx_a), *(matrix_a_ptr_block3 + col_idx_a),
      //        *(matrix_a_ptr_block4 + col_idx_a));

      __VectorLoadInput2Block1(matrix_b + col_idx_a, matrix_b_cols);
      // printf("Filter address: %f, %f, %f\n", *(matrix_b + col_idx_a),
      //        *(matrix_b + col_idx_a + matrix_b_cols),  *(matrix_b + col_idx_a + 2*matrix_b_cols));

      __VectorMulAddFloatBlock4x1();
    }

    __VectorStoreOutputBlock4x1(matrix_c + matrix_b_rows * row_idx_a,
                                matrix_c + matrix_b_rows * (row_idx_a+1),
                                matrix_c + matrix_b_rows * (row_idx_a+2),
                                matrix_c + matrix_b_rows * (row_idx_a+3));
    // printf("Output: %f, %f, %f, %f\n", *(matrix_c + matrix_b_rows * row_idx_a),
    //          *(matrix_c + matrix_b_rows * (row_idx_a+1)), *(matrix_c + matrix_b_rows * (row_idx_a+2)),
    //          *(matrix_c + matrix_b_rows * (row_idx_a+3)));
  }
  if(matrix_a_diff) {
    for(int i = new_matrix_a_rows; i < matrix_a_rows; i++){
      __VectorClearOutputAccumBlock1x1();
      const float* matrix_a_ptr_block1 = matrix_a + matrix_a_cols * i;
      for(int col_idx_a = 0; col_idx_a < matrix_a_cols; col_idx_a++){

        __VectorLoadInput1Block1(matrix_a_ptr_block1 + col_idx_a);
        //printf("Input: %f,\n", *(matrix_a_ptr_block1 + col_idx_a));

        __VectorLoadInput2Block1(matrix_b + col_idx_a, matrix_b_cols);
         // printf("Filter address: %f, %f, %f\n", *(matrix_b + col_idx_a),
         //     *(matrix_b + col_idx_a + matrix_b_cols),  *(matrix_b + col_idx_a + 2*matrix_b_cols));

        __VectorMulAddFloatBlock1x1();
      }

      __VectorStoreOutputBlock1x1(matrix_c + matrix_b_rows * i);
      //          printf("Output: %f\n", *(matrix_c + matrix_b_rows * i));

    }
  }
}


void MatrixMatrixMultiplyAccumulate4x2(const float* matrix_a,
                                       int matrix_a_rows,
                                       int matrix_a_cols,
                                       const float* matrix_b,
                                       int matrix_b_rows,
                                       int matrix_b_cols,
                                       float* matrix_c) {

  // Matrix A (input) is layed out memory as BHWC
  // Matrix B (filter) is layed out in memory as FHWC
  // Matrix C (output) is layed out in memory as BHWF
  if(matrix_a_cols != matrix_b_cols) {
    fprintf(stderr, "Input and filter depth should match %d != %d"
            , matrix_a_cols, matrix_b_cols);
    exit(1);
  }

  // We know the 2*kMaxVectorLength32 <= matrix_b_rows
  // Difference will be a special case of 4x1 multiplication
  // int block_size = 2*kMaxVectorLength32;

  // int new_matrix_b_rows =  matrix_b_rows - (matrix_b_rows & (block_size - 1));
  // int matrix_b_rows_diff = matrix_b_rows & (block_size - 1);

  SetConfig(kElementWidthMax32, kMaxVectorLength32);

  int new_matrix_a_rows =  matrix_a_rows - (matrix_a_rows & (MAX_BLOCK_SIZE - 1));
  int matrix_a_diff = matrix_a_rows & (MAX_BLOCK_SIZE - 1);


  for(int row_idx_a = 0; row_idx_a < new_matrix_a_rows; row_idx_a += MAX_BLOCK_SIZE) {
    // clear the accumulator registers
    __VectorClearOutputAccumBlock4x2();

    //compute base address for rows
    const float* matrix_a_ptr_block1 = matrix_a + matrix_a_cols * row_idx_a;
    const float* matrix_a_ptr_block2 = matrix_a + matrix_a_cols * (row_idx_a+1);
    const float* matrix_a_ptr_block3 = matrix_a + matrix_a_cols * (row_idx_a+2);
    const float* matrix_a_ptr_block4 = matrix_a + matrix_a_cols * (row_idx_a+3);

    for(int col_idx_a = 0; col_idx_a < matrix_a_cols; col_idx_a++){

      __VectorLoadInput1Block4(matrix_a_ptr_block1 + col_idx_a,
                               matrix_a_ptr_block2 + col_idx_a,
                               matrix_a_ptr_block3 + col_idx_a,
                               matrix_a_ptr_block4 + col_idx_a);

      __VectorLoadInput2Block2(matrix_b + col_idx_a,
                               matrix_b + col_idx_a + matrix_b_cols*kMaxVectorLength32,
                               matrix_b_cols);
      __VectorMulAddFloatBlock4x2();
    }

    __VectorStoreOutputBlock4x2(matrix_c + matrix_b_rows * row_idx_a,
                                matrix_c + matrix_b_rows * (row_idx_a+1),
                                matrix_c + matrix_b_rows * (row_idx_a+2),
                                matrix_c + matrix_b_rows * (row_idx_a+3),
                                matrix_c + kMaxVectorLength32 + matrix_b_rows * row_idx_a,
                                matrix_c + kMaxVectorLength32 + matrix_b_rows * (row_idx_a+1),
                                matrix_c + kMaxVectorLength32 + matrix_b_rows * (row_idx_a+2),
                                matrix_c + kMaxVectorLength32 + matrix_b_rows * (row_idx_a+3));
  }
  // handle rows in matrix a
  if(matrix_a_diff) {

    for(int i = new_matrix_a_rows; i < matrix_a_rows; i++){
        __VectorClearOutputAccumBlock1x2();
      const float* matrix_a_ptr_block1 = matrix_a + matrix_a_rows * i;

      for(int col_idx_a = 0; col_idx_a < matrix_a_cols; col_idx_a++){

        __VectorLoadInput1Block1(matrix_a_ptr_block1 + col_idx_a);
        __VectorLoadInput2Block2(matrix_b + col_idx_a,
                                 matrix_b + col_idx_a + matrix_b_cols*kMaxVectorLength32,
                                 matrix_b_cols);
        __VectorMulAddFloatBlock1x2();
      }

      __VectorStoreOutputBlock1x2(matrix_c + matrix_b_rows * i,
                                  matrix_c + kMaxVectorLength32 + matrix_b_rows * i);

    }
  }
}


void MatrixMatrixMultiplyAccumulate4x3(const float* matrix_a,
                                       int matrix_a_rows,
                                       int matrix_a_cols,
                                       const float* matrix_b,
                                       int matrix_b_rows,
                                       int matrix_b_cols,
                                       float* matrix_c) {

  // Matrix A (input) is layed out memory as BHWC
  // Matrix B (filter) is layed out in memory as FHWC
  // Matrix C (output) is layed out in memory as BHWF
  if(matrix_a_cols != matrix_b_cols) {
    fprintf(stderr, "Input and filter depth should match %d != %d"
            , matrix_a_cols, matrix_b_cols);
    exit(1);
  }

  // We know the 3*kMaxVectorLength32 <= matrix_b_rows
  // Difference will be a special case of 4x1 multiplication
  // int block_size = 3*kMaxVectorLength32;

  // int new_matrix_b_rows =  matrix_b_rows - (matrix_b_rows & (block_size - 1));
  // int matrix_b_rows_diff = matrix_b_rows & (block_size - 1);

  SetConfig(kElementWidthMax32, kMaxVectorLength32);

  int new_matrix_a_rows =  matrix_a_rows - (matrix_a_rows & (MAX_BLOCK_SIZE - 1));
  int matrix_a_diff = matrix_a_rows & (MAX_BLOCK_SIZE - 1);


  for(int row_idx_a = 0; row_idx_a < new_matrix_a_rows; row_idx_a += MAX_BLOCK_SIZE) {
    // clear the accumulator registers
    __VectorClearOutputAccumBlock4x3();

    //compute base address for rows
    const float* matrix_a_ptr_block1 = matrix_a + matrix_a_cols * row_idx_a;
    const float* matrix_a_ptr_block2 = matrix_a + matrix_a_cols * (row_idx_a+1);
    const float* matrix_a_ptr_block3 = matrix_a + matrix_a_cols * (row_idx_a+2);
    const float* matrix_a_ptr_block4 = matrix_a + matrix_a_cols * (row_idx_a+3);

    for(int col_idx_a = 0; col_idx_a < matrix_a_cols; col_idx_a++){

      __VectorLoadInput1Block4(matrix_a_ptr_block1 + col_idx_a,
                               matrix_a_ptr_block2 + col_idx_a,
                               matrix_a_ptr_block3 + col_idx_a,
                               matrix_a_ptr_block4 + col_idx_a);

      __VectorLoadInput2Block3(matrix_b + col_idx_a,
                               matrix_b + col_idx_a + matrix_b_cols*kMaxVectorLength32,
                               matrix_b + col_idx_a + matrix_b_cols*2*kMaxVectorLength32,
                               matrix_b_cols);
      __VectorMulAddFloatBlock4x3();
    }

    __VectorStoreOutputBlock4x3(matrix_c + matrix_b_rows * row_idx_a,
                                matrix_c + matrix_b_rows * (row_idx_a+1),
                                matrix_c + matrix_b_rows * (row_idx_a+2),
                                matrix_c + matrix_b_rows * (row_idx_a+3),
                                matrix_c + kMaxVectorLength32 + matrix_b_rows * row_idx_a,
                                matrix_c + kMaxVectorLength32 + matrix_b_rows * (row_idx_a+1),
                                matrix_c + kMaxVectorLength32 + matrix_b_rows * (row_idx_a+2),
                                matrix_c + kMaxVectorLength32 + matrix_b_rows * (row_idx_a+3),
                                matrix_c + 2*kMaxVectorLength32 + matrix_b_rows * row_idx_a,
                                matrix_c + 2*kMaxVectorLength32 + matrix_b_rows * (row_idx_a+1),
                                matrix_c + 2*kMaxVectorLength32 + matrix_b_rows * (row_idx_a+2),
                                matrix_c + 2*kMaxVectorLength32 + matrix_b_rows * (row_idx_a+3));
  }
  if(matrix_a_diff) {
    for(int i = new_matrix_a_rows; i < matrix_a_rows; i++){
      __VectorClearOutputAccumBlock1x3();
      const float* matrix_a_ptr_block1 = matrix_a + matrix_a_cols * i;

      for(int col_idx_a = 0; col_idx_a < matrix_a_cols; col_idx_a++){

        __VectorLoadInput1Block1(matrix_a_ptr_block1 + col_idx_a);
        __VectorLoadInput2Block3(matrix_b + col_idx_a,
                                 matrix_b + col_idx_a + matrix_b_cols*kMaxVectorLength32,
                                 matrix_b + col_idx_a + matrix_b_cols*2*kMaxVectorLength32,
                                 matrix_b_cols);
        __VectorMulAddFloatBlock1x3();
      }

      __VectorStoreOutputBlock1x3(matrix_c + matrix_b_rows * i,
                                  matrix_c + kMaxVectorLength32 + matrix_b_rows * i,
                                  matrix_c + 2*kMaxVectorLength32 + matrix_b_rows * i);

    }
  }
}

void MatrixMatrixMultiplyAccumulate(const float* matrix_a,
                                    int matrix_a_rows,
                                    int matrix_a_cols,
                                    const float* matrix_b,
                                    int matrix_b_rows,
                                    int matrix_b_cols,
                                    float* matrix_c) {

  // Matrix A (input) is layed out memory as BHWC
  // Matrix B (filter) is layed out in memory as FHWC
  // Matrix C (output) is layed out in memory as BHWF
  if(matrix_a_cols != matrix_b_cols) {
    fprintf(stderr, "Input and filter depth should match %d != %d"
            , matrix_a_cols, matrix_b_cols);
    exit(1);
  }

  // This will be the general implementation
  int block_size = MAX_BLOCK_SIZE * kMaxVectorLength32;

  int new_matrix_b_rows =  matrix_b_rows - (matrix_b_rows & (block_size - 1));
  int matrix_b_rows_diff = matrix_b_rows & (block_size - 1);

  SetConfig(kElementWidthMax32, kMaxVectorLength32);

  int new_matrix_a_rows =  matrix_a_rows - (matrix_a_rows & (MAX_BLOCK_SIZE - 1));
  int matrix_a_diff = matrix_a_rows & (MAX_BLOCK_SIZE - 1);

  for(int row_idx_b =0; row_idx_b < new_matrix_b_rows; row_idx_b += block_size){
    const float* matrix_b_ptr = matrix_b + matrix_b_cols * row_idx_b;
    float* matrix_c_ptr =  matrix_c + row_idx_b;

    for(int row_idx_a = 0; row_idx_a < new_matrix_a_rows; row_idx_a += MAX_BLOCK_SIZE) {
      // clear the accumulator registers
      __VectorClearOutputAccumBlock4x4();
      //compute base address for rows
      const float* matrix_a_ptr_block1 = matrix_a + matrix_a_cols * row_idx_a;
      const float* matrix_a_ptr_block2 = matrix_a + matrix_a_cols * (row_idx_a+1);
      const float* matrix_a_ptr_block3 = matrix_a + matrix_a_cols * (row_idx_a+2);
      const float* matrix_a_ptr_block4 = matrix_a + matrix_a_cols * (row_idx_a+3);

      for(int col_idx_a = 0; col_idx_a < matrix_a_cols; col_idx_a++){

        __VectorLoadInput1Block4(matrix_a_ptr_block1 + col_idx_a,
                                 matrix_a_ptr_block2 + col_idx_a,
                                 matrix_a_ptr_block3 + col_idx_a,
                                 matrix_a_ptr_block4 + col_idx_a);

        __VectorLoadInput2Block4(matrix_b_ptr + col_idx_a,
                                 matrix_b_ptr + col_idx_a + matrix_b_cols*kMaxVectorLength32,
                                 matrix_b_ptr + col_idx_a + matrix_b_cols*2*kMaxVectorLength32,
                                 matrix_b_ptr + col_idx_a + matrix_b_cols*3*kMaxVectorLength32,
                                 matrix_b_cols);
        __VectorMulAddFloatBlock4x4();
      }

      __VectorStoreOutputBlock4x4(matrix_c_ptr + matrix_b_rows * row_idx_a,
                                  matrix_c_ptr + matrix_b_rows * (row_idx_a+1),
                                  matrix_c_ptr + matrix_b_rows * (row_idx_a+2),
                                  matrix_c_ptr + matrix_b_rows * (row_idx_a+3),
                                  matrix_c_ptr + kMaxVectorLength32 + matrix_b_rows * row_idx_a,
                                  matrix_c_ptr + kMaxVectorLength32 + matrix_b_rows * (row_idx_a+1),
                                  matrix_c_ptr + kMaxVectorLength32 + matrix_b_rows * (row_idx_a+2),
                                  matrix_c_ptr + kMaxVectorLength32 + matrix_b_rows * (row_idx_a+3),
                                  matrix_c_ptr + 2*kMaxVectorLength32 + matrix_b_rows * row_idx_a,
                                  matrix_c_ptr + 2*kMaxVectorLength32 + matrix_b_rows * (row_idx_a+1),
                                  matrix_c_ptr + 2*kMaxVectorLength32 + matrix_b_rows * (row_idx_a+2),
                                  matrix_c_ptr + 2*kMaxVectorLength32 + matrix_b_rows * (row_idx_a+3),
                                  matrix_c_ptr + 3*kMaxVectorLength32 + matrix_b_rows * row_idx_a,
                                  matrix_c_ptr + 3*kMaxVectorLength32 + matrix_b_rows * (row_idx_a+1),
                                  matrix_c_ptr + 3*kMaxVectorLength32 + matrix_b_rows * (row_idx_a+2),
                                  matrix_c_ptr + 3*kMaxVectorLength32 + matrix_b_rows * (row_idx_a+3));
    }
    if(matrix_a_diff) {
      for(int i = new_matrix_a_rows; i < matrix_a_rows; i++){
        __VectorClearOutputAccumBlock1x4();
        const float* matrix_a_ptr_block1 = matrix_a + matrix_a_cols * i;

        for(int col_idx_a = 0; col_idx_a < matrix_a_cols; col_idx_a++){

          __VectorLoadInput1Block1(matrix_a_ptr_block1 + col_idx_a);
          __VectorLoadInput2Block4(matrix_b_ptr + col_idx_a,
                                   matrix_b_ptr + col_idx_a + matrix_b_cols*kMaxVectorLength32,
                                   matrix_b_ptr + col_idx_a + matrix_b_cols*2*kMaxVectorLength32,
                                   matrix_b_ptr + col_idx_a + matrix_b_cols*3*kMaxVectorLength32,
                                   matrix_b_cols);
          __VectorMulAddFloatBlock1x4();
        }

        __VectorStoreOutputBlock1x4(matrix_c_ptr + matrix_b_rows * i,
                                    matrix_c_ptr + kMaxVectorLength32 + matrix_b_rows * i,
                                    matrix_c_ptr + 2*kMaxVectorLength32 + matrix_b_rows * i,
                                    matrix_c_ptr + 3*kMaxVectorLength32 + matrix_b_rows * i);
      }
    }
  }
}

#endif
