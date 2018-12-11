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

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <limits>

#include "tensorflow/contrib/lite/c/builtin_op_data.h"
#include "tensorflow/contrib/lite/c/c_api_internal.h"
#include "tensorflow/contrib/lite/kernels/activation_functor.h"
#include "tensorflow/contrib/lite/experimental/riscv/kernels/reference/fully_connected_float.h"
#include "tensorflow/contrib/lite/experimental/riscv/kernels/kernel_util.h"
#include "tensorflow/contrib/lite/kernels/internal/tensor.h"
#include "tensorflow/contrib/lite/kernels/op_macros.h"

namespace tflite {
namespace ops {
namespace riscv {
namespace fully_connected {

// This file has two implementations of FullyConnected
enum KernelType {
  kReference,
  kOptimized,
};

struct OpData {
  // The scaling factor from input to output (aka the 'real multiplier') can
  // be represented as a fixed point multiplier plus a left shift.
  int32_t output_multiplier;
  int output_shift;
  // The range of the fused activation layer. For example for kNone and
  // uint8_t these would be 0 and 255.
  int32_t output_activation_min;
  int32_t output_activation_max;
  // The index of the temporary tensor where the quantized inputs are cached.
  int input_quantized_index;
};

constexpr int kInputTensor = 0;
constexpr int kWeightsTensor = 1;
constexpr int kBiasTensor = 2;
constexpr int kOutputTensor = 0;
constexpr int kShuffledInputWorkspaceTensor = 1;
constexpr int kScratchBufferTensor = 1;

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  // This is a builtin op, so we don't use the contents in 'buffer', if any.
  // Instead, we allocate a new object to carry information from Prepare() to
  // Eval().
  auto* op_data = new OpData();
  context->AddTensors(context, 1, &op_data->input_quantized_index);
  return op_data;
}

void Free(TfLiteContext* context, void* buffer) {
  delete reinterpret_cast<OpData*>(buffer);
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  auto* params =
      reinterpret_cast<TfLiteFullyConnectedParams*>(node->builtin_data);
  OpData* data = reinterpret_cast<OpData*>(node->user_data);

  // Check we have all the inputs and outputs we need.
  TF_LITE_ENSURE_EQ(context, node->inputs->size, 3);
  // Shuffled formats need a workspace to store the shuffled input activations.
  const int expected_outputs_count =
      params->weights_format == kTfLiteFullyConnectedWeightsFormatDefault ? 1
                                                                          : 2;
  TF_LITE_ENSURE_EQ(context, node->outputs->size, expected_outputs_count);

  const TfLiteTensor* input = GetInput(context, node, kInputTensor);
  const TfLiteTensor* filter = GetInput(context, node, kWeightsTensor);
  const TfLiteTensor* bias = GetOptionalInputTensor(context, node, kBiasTensor);
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);

  // Check all the parameters of tensor match within themselves and match the
  // input configuration.
  int input_size = 1;
  for (int i = 0; i < input->dims->size; i++) {
    input_size *= input->dims->data[i];
  }

  TF_LITE_ENSURE_EQ(context, NumDimensions(filter), 2);
  const int batch_size = input_size / filter->dims->data[1];
  const int num_units = filter->dims->data[0];

  TF_LITE_ENSURE_EQ(context, input_size, batch_size * filter->dims->data[1]);
  if (bias) {
    TF_LITE_ENSURE_EQ(context, NumElements(bias), SizeOfDimension(filter, 0));
  }

  // Resize output.
  TfLiteIntArray* output_size_array = TfLiteIntArrayCreate(2);
  output_size_array->data[0] = batch_size;
  output_size_array->data[1] = num_units;
  TF_LITE_ENSURE_OK(context,
                    context->ResizeTensor(context, output, output_size_array));
  return kTfLiteOk;
}

template <KernelType kernel_type>
TfLiteStatus EvalFloat(TfLiteContext* context, TfLiteNode* node,
                       TfLiteFullyConnectedParams* params, OpData* data,
                       const TfLiteTensor* input, const TfLiteTensor* filter,
                       const TfLiteTensor* bias, TfLiteTensor* output) {
  float output_activation_min, output_activation_max;
  CalculateActivationRange(params->activation, &output_activation_min,
                           &output_activation_max);
#define TF_LITE_FULLY_CONNECTED(type)                                         \
  {                                                                           \
    FullyConnectedParams op_params;                                           \
    op_params.float_activation_min = output_activation_min;                   \
    op_params.float_activation_max = output_activation_max;                   \
    type::FullyConnected(op_params, GetTensorShape(input),                    \
                         GetTensorData<float>(input), GetTensorShape(filter), \
                         GetTensorData<float>(filter), GetTensorShape(bias),  \
                         GetTensorData<float>(bias), GetTensorShape(output),  \
                         GetTensorData<float>(output));                       \
  }
  if (kernel_type == kReference) {
    TF_LITE_FULLY_CONNECTED(reference_ops);
  } else {
    static_assert("Optimized ops for RISCV not implemented yet.");
  }
#undef TF_LITE_FULLY_CONNECTED

  return kTfLiteOk;
}


template <KernelType kernel_type>
TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  auto* params =
      reinterpret_cast<TfLiteFullyConnectedParams*>(node->builtin_data);
  OpData* data = reinterpret_cast<OpData*>(node->user_data);

  const TfLiteTensor* input = GetInput(context, node, kInputTensor);
  const TfLiteTensor* filter = GetInput(context, node, kWeightsTensor);
  const TfLiteTensor* bias = GetOptionalInputTensor(context, node, kBiasTensor);
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);

  switch (filter->type) {  // Already know in/out types are same.
    case kTfLiteFloat32:
      return EvalFloat<kernel_type>(context, node, params, data, input, filter,
                                    bias, output);
    default:
      context->ReportError(context, "Type %d not currently supported.",
                           filter->type);
      return kTfLiteError;
  }
  return kTfLiteOk;
}

}  // namespace fully_connected

TfLiteRegistration* Register_FULLY_CONNECTED_REF() {
  static TfLiteRegistration r = {
      fully_connected::Init, fully_connected::Free, fully_connected::Prepare,
      fully_connected::Eval<fully_connected::kReference>};
  return &r;
}

TfLiteRegistration* Register_FULLY_CONNECTED_OPT() {
  static TfLiteRegistration r = {
      fully_connected::Init, fully_connected::Free, fully_connected::Prepare,
      fully_connected::Eval<fully_connected::kOptimized>};
  return &r;
}

 TfLiteRegistration* Register_FULLY_CONNECTED() {
  #ifdef RISCV
    return Register_FULLY_CONNECTED_OPT();
  #else
    return Register_FULLY_CONNECTED_REF();
  #endif
}

}  // namespace riscv
}  // namespace ops
}  // namespace tflite
