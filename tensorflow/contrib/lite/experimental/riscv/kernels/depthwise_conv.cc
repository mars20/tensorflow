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
#include "tensorflow/contrib/lite/experimental/riscv/kernels/reference/depthwiseconv_float.h"
#include "tensorflow/contrib/lite/experimental/riscv/kernels/optimized/depthwiseconv_float.h"
#include "tensorflow/contrib/lite/experimental/riscv/kernels/kernel_util.h"
#include "tensorflow/contrib/lite/kernels/internal/tensor.h"
#include "tensorflow/contrib/lite/kernels/op_macros.h"
#include "tensorflow/contrib/lite/kernels/padding.h"
#include "tensorflow/contrib/lite/experimental/riscv/profiling/stats.h"

namespace tflite {
namespace ops {
namespace riscv {
namespace depthwise_conv {

constexpr int kInputTensor = 0;
constexpr int kFilterTensor = 1;
constexpr int kBiasTensor = 2;
constexpr int kOutputTensor = 0;

// This file has two implementation of DepthwiseConv.
enum KernelType {
  kReference,
  kOptimized,
};

struct OpData {
  TfLitePaddingValues padding;
  // The scaling factor from input to output (aka the 'real multiplier') can
  // be represented as a fixed point multiplier plus a left shift.
  int32_t output_multiplier;
  int output_shift;
  // The range of the fused activation layer. For example for kNone and
  // uint8_t these would be 0 and 255.
  int32_t output_activation_min;
  int32_t output_activation_max;
};

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  // This is a builtin op, so we don't use the contents in 'buffer', if any.
  // Instead, we allocate a new object to carry information from Prepare() to
  // Eval().
  return new OpData;
}

void Free(TfLiteContext* context, void* buffer) {
  delete reinterpret_cast<OpData*>(buffer);
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  auto* params =
      reinterpret_cast<TfLiteDepthwiseConvParams*>(node->builtin_data);
  OpData* data = reinterpret_cast<OpData*>(node->user_data);

  // TODO(ahentz): use could use GetOptionalInputTensor() here, but we need to
  // decide whether we are OK with optional tensors being completely absent, as
  // opposed to having -1 as their index.
  bool hasBias = NumInputs(node) == 3;

  TF_LITE_ENSURE(context, hasBias || NumInputs(node) == 2);
  const TfLiteTensor* input = GetInput(context, node, kInputTensor);
  const TfLiteTensor* filter = GetInput(context, node, kFilterTensor);
  const TfLiteTensor* bias = nullptr;

  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);

  TF_LITE_ENSURE_EQ(context, NumDimensions(input), 4);
  TF_LITE_ENSURE_EQ(context, NumDimensions(filter), 4);

  // The parameter 'depth_multiplier' is redundant, so we check here to make
  // sure it is consistent with the given dimensions.
  TF_LITE_ENSURE_EQ(context,
                    params->depth_multiplier * SizeOfDimension(input, 3),
                    SizeOfDimension(filter, 3));

  const TfLiteType data_type = input->type;
  TF_LITE_ENSURE(context, data_type == kTfLiteFloat32);
  TF_LITE_ENSURE_EQ(context, output->type, data_type);
  TF_LITE_ENSURE_EQ(context, filter->type, data_type);

  if (hasBias) {
    bias = GetInput(context, node, kBiasTensor);
    TF_LITE_ENSURE_EQ(context, bias->type, data_type);
    TF_LITE_ENSURE_EQ(context, NumDimensions(bias), 1);
    TF_LITE_ENSURE_EQ(context, SizeOfDimension(filter, 3),
                      SizeOfDimension(bias, 0));
  }

  int channels_out = SizeOfDimension(filter, 3);
  int width = SizeOfDimension(input, 2);
  int height = SizeOfDimension(input, 1);
  int filter_width = SizeOfDimension(filter, 2);
  int filter_height = SizeOfDimension(filter, 1);
  int batches = SizeOfDimension(input, 0);

  // Matching GetWindowedOutputSize in TensorFlow.
  auto padding = params->padding;
  auto compute_out_size = [padding](int image_size, int filter_size, int stride,
                                    int dilation_rate) -> int {
    int effective_filter_size = (filter_size - 1) * dilation_rate + 1;
    return padding == kTfLitePaddingSame
               ? (image_size + stride - 1) / stride
               : padding == kTfLitePaddingValid
                     ? (image_size - effective_filter_size + stride) / stride
                     : 0;
  };

  int out_width = compute_out_size(width, filter_width, params->stride_width,
                                   params->dilation_width_factor);
  int out_height =
      compute_out_size(height, filter_height, params->stride_height,
                       params->dilation_height_factor);

  data->padding.height =
      ComputePadding(params->stride_height, params->dilation_height_factor,
                     height, filter_height, out_height);
  data->padding.width =
      ComputePadding(params->stride_width, params->dilation_width_factor, width,
                     filter_width, out_width);

  TfLiteIntArray* outputSize = TfLiteIntArrayCreate(4);
  outputSize->data[0] = batches;
  outputSize->data[1] = out_height;
  outputSize->data[2] = out_width;
  outputSize->data[3] = channels_out;
  return context->ResizeTensor(context, output, outputSize);
}

template <KernelType kernel_type>
void EvalFloat(TfLiteContext* context, TfLiteNode* node,
               TfLiteDepthwiseConvParams* params, OpData* data,
               const TfLiteTensor* input, const TfLiteTensor* filter,
               const TfLiteTensor* bias, TfLiteTensor* output) {
  float output_activation_min, output_activation_max;
  CalculateActivationRange(params->activation, &output_activation_min,
                           &output_activation_max);

  void (*depthwise_conv)(const DepthwiseParams&, const RuntimeShape&,
                         const float*, const RuntimeShape&, const float*,
                         const RuntimeShape&, const float*, const RuntimeShape&,
                         float*);
  if (kernel_type == kReference) {
    depthwise_conv = &reference_ops::DepthwiseConv;
  } else if(kernel_type == kOptimized){
    #ifdef RISCV
     depthwise_conv = &optimized_ops::DepthwiseConv;
    #endif
  }else {
    static_assert("Optimized ops for RISCV not implemented yet.");
  }

  DepthwiseParams op_params;
  op_params.padding_type = PaddingType::kSame;
  op_params.padding_values.width = data->padding.width;
  op_params.padding_values.height = data->padding.height;
  op_params.stride_width = params->stride_width;
  op_params.stride_height = params->stride_height;
  op_params.dilation_width_factor = params->dilation_width_factor;
  op_params.dilation_height_factor = params->dilation_height_factor;
  op_params.depth_multiplier = params->depth_multiplier;
  op_params.float_activation_min = output_activation_min;
  op_params.float_activation_max = output_activation_max;
  // printf("=== Depthwise Conv ===\n");
  // tflite::riscv::stats::csr counters_depthconv;
  // tflite::riscv::stats::StartStats(&counters_depthconv);
  depthwise_conv(op_params, GetTensorShape(input), GetTensorData<float>(input),
                 GetTensorShape(filter), GetTensorData<float>(filter),
                 GetTensorShape(bias), GetTensorData<float>(bias),
                 GetTensorShape(output), GetTensorData<float>(output));
  // tflite::riscv::stats::StopStats(&counters_depthconv);
  // tflite::riscv::stats::PrintStats(&counters_depthconv);
}

template <KernelType kernel_type>
TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  auto* params =
      reinterpret_cast<TfLiteDepthwiseConvParams*>(node->builtin_data);
  OpData* data = reinterpret_cast<OpData*>(node->user_data);

  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);
  const TfLiteTensor* input = GetInput(context, node, kInputTensor);
  const TfLiteTensor* filter = GetInput(context, node, kFilterTensor);
  const TfLiteTensor* bias =
      (NumInputs(node) == 3) ? GetInput(context, node, kBiasTensor) : nullptr;

  // TODO(aselle): Consider whether float conv and quantized conv should be
  // separate ops to avoid dispatch overhead here.
  switch (input->type) {  // Already know in/out types are same.
    case kTfLiteFloat32:
      EvalFloat<kernel_type>(context, node, params, data, input, filter, bias,
                             output);
      break;
    default:
      context->ReportError(context, "Type %d not currently supported.",
                           input->type);
      return kTfLiteError;
  }
  return kTfLiteOk;
}

}  // namespace depthwise_conv

TfLiteRegistration* Register_DEPTHWISE_CONVOLUTION_REF() {
  static TfLiteRegistration r = {
      depthwise_conv::Init, depthwise_conv::Free, depthwise_conv::Prepare,
      depthwise_conv::Eval<depthwise_conv::kReference>};
  return &r;
}

TfLiteRegistration* Register_DEPTHWISE_CONVOLUTION_OPT() {
  static TfLiteRegistration r = {
      depthwise_conv::Init, depthwise_conv::Free, depthwise_conv::Prepare,
      depthwise_conv::Eval<depthwise_conv::kOptimized>};
  return &r;
}

TfLiteRegistration* Register_DEPTHWISE_CONV_2D() {
  #ifdef RISCV
    return Register_DEPTHWISE_CONVOLUTION_OPT();
  #else
    return Register_DEPTHWISE_CONVOLUTION_REF();
  #endif
}

}  // namespace riscv
}  // namespace ops
}  // namespace tflite
