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
#ifndef TENSORFLOW_CONTRIB_LITE_EXPERIMENTAL_RISCV_KERNELS_OPTIMIZED_LSTM_FLOAT_H_
#define TENSORFLOW_CONTRIB_LITE_EXPERIMENTAL_RISCV_KERNELS_OPTIMIZED_LSTM_FLOAT_H_

#include <vector>
#include "tensorflow/contrib/lite/experimental/riscv/kernels/common.h"
#include "tensorflow/contrib/lite/experimental/riscv/kernels/optimized/fully_connected_float.h"
#include "tensorflow/contrib/lite/experimental/riscv/kernels/optimized/intrinsic/riscv_ml_extension.h"
#include "tensorflow/contrib/lite/experimental/riscv/kernels/optimized/optimized_ops_float.h"
#include "tensorflow/contrib/lite/experimental/riscv/kernels/reference/portable_tensor_utils.h"
#include "tensorflow/contrib/lite/experimental/riscv/kernels/reference/reference_ops_float.h"
#include "tensorflow/contrib/lite/kernels/internal/types.h"
#ifdef RISCV
namespace tflite {
namespace optimized_ops {

// Performs an LSTM batch inference step for input specified by input_ptr_batch.
// The LSTM cell is specified by the pointers to its weights (*_weights_ptr) and
// biases (*_bias_ptr), and buffers (*_scratch), along with additional
// parameters:
//  - params: various LSTM params including activation, clipping, etc.,
//  - n_batch: size of batch,
//  - n_cell: number of cells (or units),
//  - n_input: the input size,
//  - n_output: the output size.
//  - output_batch_leading_dim: the leading dimension of the output buffer.
//
// The pointers to the cell and output state and the output are updated.
//
// The pointers with the suffix "_batch" point to data aligned in batch_major
// order, and each step processes batch_size many inputs from input_ptr_batch,
// and updates batch_size many cell and output states.
//
// The output_batch_dim is output.shape[-1], i.e. the outermost dimension of the
// output tensor, and in most cases will be equal to n_output. It is usually not
// when we want to store the LSTM output into a slice of the output tensor, e.g.
// for bidirectional LSTMs with merge_outputs. In this case, the batched
// operations cannot be used since they assume that the batched outputs are
// contiguous, and we manually loop over the batched outputs.
inline void LstmStepWithAuxInput(
    const float* input_ptr_batch, const float* input_to_input_weights_ptr,
    const float* input_to_forget_weights_ptr,
    const float* input_to_cell_weights_ptr,
    const float* input_to_output_weights_ptr, const float* aux_input_ptr_batch,
    const float* aux_input_to_input_weights_ptr,
    const float* aux_input_to_forget_weights_ptr,
    const float* aux_input_to_cell_weights_ptr,
    const float* aux_input_to_output_weights_ptr,
    const float* recurrent_to_input_weights_ptr,
    const float* recurrent_to_forget_weights_ptr,
    const float* recurrent_to_cell_weights_ptr,
    const float* recurrent_to_output_weights_ptr,
    const float* cell_to_input_weights_ptr,
    const float* cell_to_forget_weights_ptr,
    const float* cell_to_output_weights_ptr, const float* input_gate_bias_ptr,
    const float* forget_gate_bias_ptr, const float* cell_bias_ptr,
    const float* output_gate_bias_ptr, const float* projection_weights_ptr,
    const float* projection_bias_ptr, const TfLiteLSTMParams* params,
    int n_batch, int n_cell, int n_input, int n_aux_input, int n_output,
    int output_batch_leading_dim, float* output_state_ptr,
    float* cell_state_ptr, float* input_gate_scratch,
    float* forget_gate_scratch, float* cell_scratch, float* output_gate_scratch,
    float* output_ptr_batch) {
  // Since we have already checked that weights are all there or none, we can
  // check the existense of only one to the get the condition.
  const bool use_cifg = (input_to_input_weights_ptr == nullptr);
  const bool use_peephole = (cell_to_output_weights_ptr != nullptr);
  // Initialize scratch buffers with bias.
  if (!use_cifg) {
    tensor_utils::VectorBatchVectorAssign(input_gate_bias_ptr, n_cell, n_batch,
                                          input_gate_scratch);
  }
  tensor_utils::VectorBatchVectorAssign(forget_gate_bias_ptr, n_cell, n_batch,
                                        forget_gate_scratch);
  tensor_utils::VectorBatchVectorAssign(cell_bias_ptr, n_cell, n_batch,
                                        cell_scratch);
  tensor_utils::VectorBatchVectorAssign(output_gate_bias_ptr, n_cell, n_batch,
                                        output_gate_scratch);

  // For each batch and cell: compute input_weight * input.
  if (!use_cifg) {
    MatrixBatchVectorMultiplyAccumulate(
        input_to_input_weights_ptr, n_cell, n_input, input_ptr_batch, n_batch,
        input_gate_scratch, /*result_stride=*/1);
  }

  MatrixBatchVectorMultiplyAccumulate(
      input_to_forget_weights_ptr, n_cell, n_input, input_ptr_batch, n_batch,
      forget_gate_scratch, /*result_stride=*/1);
  MatrixBatchVectorMultiplyAccumulate(
      input_to_cell_weights_ptr, n_cell, n_input, input_ptr_batch, n_batch,
      cell_scratch, /*result_stride=*/1);
  MatrixBatchVectorMultiplyAccumulate(
      input_to_output_weights_ptr, n_cell, n_input, input_ptr_batch, n_batch,
      output_gate_scratch, /*result_stride=*/1);

  // If auxiliary input is available then compute aux_input_weight * aux_input
  if (aux_input_ptr_batch != nullptr) {
    if (!use_cifg) {
      MatrixBatchVectorMultiplyAccumulate(
          aux_input_to_input_weights_ptr, n_cell, n_aux_input,
          aux_input_ptr_batch, n_batch, input_gate_scratch,
          /*result_stride=*/1);
    }

    MatrixBatchVectorMultiplyAccumulate(
        aux_input_to_forget_weights_ptr, n_cell, n_aux_input,
        aux_input_ptr_batch, n_batch, forget_gate_scratch, /*result_stride=*/1);
    MatrixBatchVectorMultiplyAccumulate(
        aux_input_to_cell_weights_ptr, n_cell, n_aux_input, aux_input_ptr_batch,
        n_batch, cell_scratch, /*result_stride=*/1);
    MatrixBatchVectorMultiplyAccumulate(
        aux_input_to_output_weights_ptr, n_cell, n_aux_input,
        aux_input_ptr_batch, n_batch, output_gate_scratch, /*result_stride=*/1);
  }

  // For each batch and cell: compute recurrent_weight * output_state.
  if (!use_cifg) {
    MatrixBatchVectorMultiplyAccumulate(
        recurrent_to_input_weights_ptr, n_cell, n_output, output_state_ptr,
        n_batch, input_gate_scratch, /*result_stride=*/1);
  }
  MatrixBatchVectorMultiplyAccumulate(
      recurrent_to_forget_weights_ptr, n_cell, n_output, output_state_ptr,
      n_batch, forget_gate_scratch,
      /*result_stride=*/1);
  MatrixBatchVectorMultiplyAccumulate(
      recurrent_to_cell_weights_ptr, n_cell, n_output, output_state_ptr,
      n_batch, cell_scratch, /*result_stride=*/1);
  MatrixBatchVectorMultiplyAccumulate(
      recurrent_to_output_weights_ptr, n_cell, n_output, output_state_ptr,
      n_batch, output_gate_scratch,
      /*result_stride=*/1);

  // For each batch and cell: update input gate.
  if (!use_cifg) {
    if (use_peephole) {
      VectorBatchVectorCwiseProductAccumulate(
          cell_to_input_weights_ptr, n_cell, cell_state_ptr, n_batch,
          input_gate_scratch);
    }
    tensor_utils::ApplySigmoidToVector(input_gate_scratch, n_cell * n_batch,
                                       input_gate_scratch);
  }

  // For each batch and cell: update forget gate.
  if (use_peephole) {
    tensor_utils::VectorBatchVectorCwiseProductAccumulate(
        cell_to_forget_weights_ptr, n_cell, cell_state_ptr, n_batch,
        forget_gate_scratch);
  }
  tensor_utils::ApplySigmoidToVector(forget_gate_scratch, n_cell * n_batch,
                                     forget_gate_scratch);

  // For each batch and cell: update the cell.
  tensor_utils::VectorVectorCwiseProduct(forget_gate_scratch, cell_state_ptr,
                                         n_batch * n_cell, cell_state_ptr);
  tensor_utils::ApplyActivationToVector(cell_scratch, n_batch * n_cell,
                                        params->activation, cell_scratch);
  if (use_cifg) {
    tensor_utils::Sub1Vector(forget_gate_scratch, n_batch * n_cell,
                             forget_gate_scratch);
    tensor_utils::VectorVectorCwiseProductAccumulate(
        cell_scratch, forget_gate_scratch, n_batch * n_cell, cell_state_ptr);
  } else {
    tensor_utils::VectorVectorCwiseProductAccumulate(
        cell_scratch, input_gate_scratch, n_batch * n_cell, cell_state_ptr);
  }
  if (params->cell_clip > 0.0) {
    tensor_utils::ClipVector(cell_state_ptr, n_batch * n_cell,
                             params->cell_clip, cell_state_ptr);
  }

  // For each batch and cell: update the output gate.
  if (use_peephole) {
    tensor_utils::VectorBatchVectorCwiseProductAccumulate(
        cell_to_output_weights_ptr, n_cell, cell_state_ptr, n_batch,
        output_gate_scratch);
  }
  tensor_utils::ApplySigmoidToVector(output_gate_scratch, n_batch * n_cell,
                                     output_gate_scratch);
  tensor_utils::ApplyActivationToVector(cell_state_ptr, n_batch * n_cell,
                                        params->activation, cell_scratch);
  tensor_utils::VectorVectorCwiseProduct(output_gate_scratch, cell_scratch,
                                         n_batch * n_cell, output_gate_scratch);

  const bool use_projection_weight = (projection_weights_ptr != nullptr);
  const bool use_projection_bias = (projection_bias_ptr != nullptr);

  // For each batch: update the projection and output_state. Note that since
  // the output batch rows may not be contiguous (output_batch_leading_dim !=
  // n_output), we unroll the batched operations where this is the case.
  if (output_batch_leading_dim == n_output) {
    if (use_projection_weight) {
      if (use_projection_bias) {
        tensor_utils::VectorBatchVectorAssign(projection_bias_ptr, n_output,
                                              n_batch, output_ptr_batch);
      } else {
        tensor_utils::ZeroVector(output_ptr_batch, n_batch * n_output);
      }
      tensor_utils::MatrixBatchVectorMultiplyAccumulate(
          projection_weights_ptr, n_output, n_cell, output_gate_scratch,
          n_batch, output_ptr_batch, /*result_stride=*/1);
      if (params->proj_clip > 0.0) {
        tensor_utils::ClipVector(output_ptr_batch, n_batch * n_output,
                                 params->proj_clip, output_ptr_batch);
      }
    } else {
      tensor_utils::CopyVector(output_gate_scratch, n_batch * n_output,
                               output_ptr_batch);
    }
    tensor_utils::CopyVector(output_ptr_batch, n_batch * n_output,
                             output_state_ptr);
  } else {
    if (use_projection_weight) {
      if (use_projection_bias) {
        for (int k = 0; k < n_batch; k++) {
          tensor_utils::CopyVector(
              projection_bias_ptr, n_output,
              output_ptr_batch + k * output_batch_leading_dim);
        }
      } else {
        for (int k = 0; k < n_batch; k++) {
          tensor_utils::ZeroVector(
              output_ptr_batch + k * output_batch_leading_dim, n_output);
        }
      }
      for (int k = 0; k < n_batch; k++) {
        tensor_utils::MatrixBatchVectorMultiplyAccumulate(
            projection_weights_ptr, n_output, n_cell,
            output_gate_scratch + k * n_cell,
            /*n_batch=*/1, output_ptr_batch + k * output_batch_leading_dim,
            /*result_stride=*/1);
        if (params->proj_clip > 0.0) {
          tensor_utils::ClipVector(
              output_ptr_batch + k * output_batch_leading_dim, n_output,
              params->proj_clip,
              output_ptr_batch + k * output_batch_leading_dim);
        }
      }
    } else {
      for (int k = 0; k < n_batch; k++) {
        tensor_utils::CopyVector(
            output_gate_scratch + k * n_output, n_output,
            output_ptr_batch + k * output_batch_leading_dim);
      }
    }
    for (int k = 0; k < n_batch; k++) {
      tensor_utils::CopyVector(output_ptr_batch + k * output_batch_leading_dim,
                               n_output, output_state_ptr + k * n_output);
    }
  }
}

inline void FullLstmCell(
    const TfLiteTensor* input, const TfLiteTensor* input_to_input_weights,
    const TfLiteTensor* input_to_forget_weights,
    const TfLiteTensor* input_to_cell_weights,
    const TfLiteTensor* input_to_output_weights,
    const TfLiteTensor* recurrent_to_input_weights,
    const TfLiteTensor* recurrent_to_forget_weights,
    const TfLiteTensor* recurrent_to_cell_weights,
    const TfLiteTensor* recurrent_to_output_weights,
    const TfLiteTensor* cell_to_input_weights,
    const TfLiteTensor* cell_to_forget_weights,
    const TfLiteTensor* cell_to_output_weights, const TfLiteTensor* aux_input,
    const TfLiteTensor* aux_input_to_input_weights,
    const TfLiteTensor* aux_input_to_forget_weights,
    const TfLiteTensor* aux_input_to_cell_weights,
    const TfLiteTensor* aux_input_to_output_weights,
    const TfLiteTensor* input_gate_bias, const TfLiteTensor* forget_gate_bias,
    const TfLiteTensor* cell_bias, const TfLiteTensor* output_gate_bias,
    const TfLiteTensor* projection_weights, const TfLiteTensor* projection_bias,
    const TfLiteLSTMParams* params, bool forward_sequence, int output_offset,
    TfLiteTensor* scratch_buffer, TfLiteTensor* activation_state,
    TfLiteTensor* cell_state, TfLiteTensor* output) {
  TF_LITE_ASSERT(input->dims->size >= 2 && input->dims->size <= 3);
  const int max_time = (input->dims->size == 2) ? 1 : input->dims->data[0];
  const int n_batch = input->dims->data[input->dims->size - 2];
  const int n_input = input->dims->data[input->dims->size - 1];
  const int aux_input_size =
      (aux_input) ? aux_input->dims->data[aux_input->dims->size - 1] : 0;

  // n_cell and n_output will be the same size when there is no projection.
  const int n_cell = input_to_output_weights->dims->data[0];
  const int n_output = recurrent_to_output_weights->dims->data[1];

  // Since we have already checked that weights are all there or none, we can
  // check the existense of only one to the get the condition.
  const bool use_cifg = (input_to_input_weights == nullptr);
  const bool use_peephole = (cell_to_output_weights != nullptr);

  // Index the scratch buffers pointers to the global scratch buffer.
  float* input_gate_scratch = nullptr;
  float* cell_scratch = nullptr;
  float* forget_gate_scratch = nullptr;
  float* output_gate_scratch = nullptr;
  if (use_cifg) {
    cell_scratch = scratch_buffer->data.f;
    forget_gate_scratch = scratch_buffer->data.f + n_cell * n_batch;
    output_gate_scratch = scratch_buffer->data.f + 2 * n_cell * n_batch;
  } else {
    input_gate_scratch = scratch_buffer->data.f;
    cell_scratch = scratch_buffer->data.f + n_cell * n_batch;
    forget_gate_scratch = scratch_buffer->data.f + 2 * n_cell * n_batch;
    output_gate_scratch = scratch_buffer->data.f + 3 * n_cell * n_batch;
  }

  // Check optional tensors, the respective pointers can be null.
  const float* input_to_input_weights_ptr =
      (use_cifg) ? nullptr : input_to_input_weights->data.f;
  const float* recurrent_to_input_weights_ptr =
      (use_cifg) ? nullptr : recurrent_to_input_weights->data.f;
  const float* input_gate_bias_ptr =
      (use_cifg) ? nullptr : input_gate_bias->data.f;
  const float* cell_to_input_weights_ptr =
      (use_peephole && !use_cifg) ? cell_to_input_weights->data.f : nullptr;
  const float* cell_to_forget_weights_ptr =
      (use_peephole) ? cell_to_forget_weights->data.f : nullptr;
  const float* cell_to_output_weights_ptr =
      (use_peephole) ? cell_to_output_weights->data.f : nullptr;
  const float* projection_weights_ptr =
      (projection_weights == nullptr) ? nullptr : projection_weights->data.f;
  const float* projection_bias_ptr =
      (projection_bias == nullptr) ? nullptr : projection_bias->data.f;

  float* aux_input_ptr = nullptr;
  float* aux_input_to_input_weights_ptr = nullptr;
  float* aux_input_to_forget_weights_ptr = nullptr;
  float* aux_input_to_cell_weights_ptr = nullptr;
  float* aux_input_to_output_weights_ptr = nullptr;
  if (aux_input_size > 0) {
    aux_input_ptr = aux_input->data.f;
    if (!use_cifg) {
      aux_input_to_input_weights_ptr = aux_input_to_input_weights->data.f;
    }
    aux_input_to_forget_weights_ptr = aux_input_to_forget_weights->data.f;
    aux_input_to_cell_weights_ptr = aux_input_to_cell_weights->data.f;
    aux_input_to_output_weights_ptr = aux_input_to_output_weights->data.f;
  }

  // Loop through the sequence.
  const int output_batch_leading_dim =
      output->dims->data[output->dims->size - 1];
  const int input_step = n_batch * n_input;
  const int output_step = n_batch * output_batch_leading_dim;
  for (int t = 0; t < max_time; t++) {
    // If this is the forward_sequence, step forward, otherwise step backwards.
    const int t_rel = forward_sequence ? t : max_time - t - 1;
    const float* input_ptr = input->data.f + t_rel * input_step;
    float* output_ptr_time =
        output->data.f + t_rel * output_step + output_offset;

    LstmStepWithAuxInput(
        input_ptr, input_to_input_weights_ptr, input_to_forget_weights->data.f,
        input_to_cell_weights->data.f, input_to_output_weights->data.f,
        aux_input_ptr, aux_input_to_input_weights_ptr,
        aux_input_to_forget_weights_ptr, aux_input_to_cell_weights_ptr,
        aux_input_to_output_weights_ptr, recurrent_to_input_weights_ptr,
        recurrent_to_forget_weights->data.f, recurrent_to_cell_weights->data.f,
        recurrent_to_output_weights->data.f, cell_to_input_weights_ptr,
        cell_to_forget_weights_ptr, cell_to_output_weights_ptr,
        input_gate_bias_ptr, forget_gate_bias->data.f, cell_bias->data.f,
        output_gate_bias->data.f, projection_weights_ptr, projection_bias_ptr,
        params, n_batch, n_cell, n_input, aux_input_size, n_output,
        output_batch_leading_dim, activation_state->data.f, cell_state->data.f,
        input_gate_scratch, forget_gate_scratch, cell_scratch,
        output_gate_scratch, output_ptr_time);
  }
}

inline void BasicLstmCell(
    const LstmCellParams& params, const RuntimeShape& unextended_input_shape,
    const float* input_data, const RuntimeShape& unextended_prev_activ_shape,
    const float* prev_activ_data, const RuntimeShape& weights_shape,
    const float* weights_data, const RuntimeShape& unextended_bias_shape,
    const float* bias_data, const RuntimeShape& unextended_prev_state_shape,
    const float* prev_state_data,
    const RuntimeShape& unextended_output_state_shape, float* output_state_data,
    const RuntimeShape& unextended_output_activ_shape, float* output_activ_data,
    const RuntimeShape& unextended_concat_temp_shape, float* concat_temp_data,
    const RuntimeShape& unextended_activ_temp_shape, float* activ_temp_data) {
  TFLITE_DCHECK_LE(unextended_input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_LE(unextended_prev_activ_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_LE(unextended_bias_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_LE(unextended_prev_state_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_LE(unextended_output_state_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_LE(unextended_output_activ_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_LE(unextended_concat_temp_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_LE(unextended_activ_temp_shape.DimensionsCount(), 4);
  const RuntimeShape input_shape =
      RuntimeShape::ExtendedShape(4, unextended_input_shape);
  const RuntimeShape prev_activ_shape =
      RuntimeShape::ExtendedShape(4, unextended_prev_activ_shape);
  const RuntimeShape bias_shape =
      RuntimeShape::ExtendedShape(4, unextended_bias_shape);
  const RuntimeShape prev_state_shape =
      RuntimeShape::ExtendedShape(4, unextended_prev_state_shape);
  const RuntimeShape output_state_shape =
      RuntimeShape::ExtendedShape(4, unextended_output_state_shape);
  const RuntimeShape output_activ_shape =
      RuntimeShape::ExtendedShape(4, unextended_output_activ_shape);
  const RuntimeShape concat_temp_shape =
      RuntimeShape::ExtendedShape(4, unextended_concat_temp_shape);
  const RuntimeShape activ_temp_shape =
      RuntimeShape::ExtendedShape(4, unextended_activ_temp_shape);
  TFLITE_DCHECK_GE(weights_shape.DimensionsCount(), 2);

  const int weights_dim_count = weights_shape.DimensionsCount();
  const int batches =
      MatchingDim(input_shape, 0, prev_activ_shape, 0, prev_state_shape, 0,
                  output_state_shape, 0, output_activ_shape, 0);
  const int height =
      MatchingDim(input_shape, 1, prev_activ_shape, 1, prev_state_shape, 1,
                  output_state_shape, 1, output_activ_shape, 1);
  const int width =
      MatchingDim(input_shape, 2, prev_activ_shape, 2, prev_state_shape, 2,
                  output_state_shape, 2, output_activ_shape, 2);
  const int input_depth = input_shape.Dims(3);
  const int prev_activ_depth = prev_activ_shape.Dims(3);
  const int total_input_depth = prev_activ_depth + input_depth;
  TFLITE_DCHECK_EQ(weights_shape.Dims(weights_dim_count - 1),
                   total_input_depth);
  TFLITE_DCHECK_EQ(FlatSizeSkipDim(bias_shape, 3), 1);
  const int intern_activ_depth =
      MatchingDim(weights_shape, weights_dim_count - 2, bias_shape, 3);
  TFLITE_DCHECK_EQ(weights_shape.FlatSize(),
                   intern_activ_depth * total_input_depth);
  TFLITE_DCHECK_EQ(intern_activ_depth % 4, 0);
  const int output_depth =
      MatchingDim(prev_state_shape, 3, prev_activ_shape, 3, output_state_shape,
                  3, output_activ_shape, 3);
  TFLITE_DCHECK_EQ(output_depth, intern_activ_depth / 4);

  // Concatenate prev_activ and input data together
  std::vector<float const*> concat_input_arrays_data;
  std::vector<RuntimeShape const*> concat_input_arrays_shapes;
  concat_input_arrays_data.push_back(input_data);
  concat_input_arrays_data.push_back(prev_activ_data);
  concat_input_arrays_shapes.push_back(&input_shape);
  concat_input_arrays_shapes.push_back(&prev_activ_shape);
  tflite::ConcatenationParams concat_params;
  concat_params.axis = 3;
  concat_params.inputs_count = concat_input_arrays_data.size();
  tflite::reference_ops::Concatenation(concat_params, &(concat_input_arrays_shapes[0]),
                &(concat_input_arrays_data[0]), concat_temp_shape,
                concat_temp_data);

  // Fully connected
  tflite::FullyConnectedParams fc_params;
  fc_params.float_activation_min = std::numeric_limits<float>::lowest();
  fc_params.float_activation_max = std::numeric_limits<float>::max();
  tflite::optimized_ops::FullyConnected(fc_params, concat_temp_shape, concat_temp_data, weights_shape,
                 weights_data, bias_shape, bias_data, activ_temp_shape,
                 activ_temp_data);

  // Memory state update (the LSTM "guts")
  for (int b = 0; b < batches; ++b) {
    for (int w = 0; w < width; ++w) {
      for (int h = 0; h < height; ++h) {
        for (int c = 0; c < output_depth; ++c) {
          const float input_gate =
              1.f /
              (1.f + std::exp(-activ_temp_data[Offset(activ_temp_shape, b, h, w,
                                                      0 * output_depth + c)]));
          const float new_input = std::tanh(activ_temp_data[Offset(
              activ_temp_shape, b, h, w, 1 * output_depth + c)]);
          const float forget_gate =
              1.f /
              (1.f + std::exp(-activ_temp_data[Offset(activ_temp_shape, b, h, w,
                                                      2 * output_depth + c)]));
          const float output_gate =
              1.f /
              (1.f + std::exp(-activ_temp_data[Offset(activ_temp_shape, b, h, w,
                                                      3 * output_depth + c)]));
          const float new_state =
              input_gate * new_input +
              forget_gate *
                  prev_state_data[Offset(prev_state_shape, b, h, w, c)];
          output_state_data[Offset(output_state_shape, b, h, w, c)] = new_state;
          output_activ_data[Offset(output_activ_shape, b, h, w, c)] =
              output_gate * std::tanh(new_state);
        }
      }
    }
  }
}

}  // namespace optimized_ops
}  // namespace tflite
#endif
#endif  // TENSORFLOW_CONTRIB_LITE_EXPERIMENTAL_RISCV_KERNELS_OPTIMIZED_LSTM_FLOAT_H_
