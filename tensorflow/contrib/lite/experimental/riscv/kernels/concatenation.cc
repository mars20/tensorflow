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
#include "tensorflow/contrib/lite/experimental/riscv/kernels/reference/reference_ops_float.h"
#include "tensorflow/contrib/lite/experimental/riscv/kernels/kernel_util.h"
#include "tensorflow/contrib/lite/kernels/internal/tensor.h"
#include "tensorflow/contrib/lite/kernels/kernel_util.h"
#include "tensorflow/contrib/lite/kernels/op_macros.h"

namespace tflite {
namespace ops {
namespace riscv {
namespace concatenation {

// This file has two implementation of Concatenation.
enum KernelType {
  kReference,
  kOptimized,
};

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  auto* params =
      reinterpret_cast<TfLiteConcatenationParams*>(node->builtin_data);
  int axis = params->axis;
  int num_inputs = node->inputs->size;

  // The number of dimensions of the input tensors must match, and all
  // dimensions except 'axis' must be equal.
  TfLiteTensor* t0 = &context->tensors[node->inputs->data[0]];
  TfLiteType input_type = t0->type;
  if (axis < 0) axis += t0->dims->size;
  TF_LITE_ENSURE(context, axis >= 0);
  TF_LITE_ENSURE(context, axis < t0->dims->size);

  // TODO(ahentz): These are limitations of our implementation that could be
  // removed with a bit of effort.
  TF_LITE_ENSURE(context, t0->dims->size <= 4);
  TF_LITE_ENSURE_EQ(context, params->activation, kTfLiteActNone);
  TF_LITE_ENSURE(context,
                 input_type == kTfLiteFloat32 || input_type == kTfLiteUInt8 ||
                     input_type == kTfLiteInt16 || input_type == kTfLiteInt32 ||
                     input_type == kTfLiteInt64);

  // Output dimensions will match input dimensions, except 'axis', which
  // will be the sum of inputs
  int sum_axis = t0->dims->data[axis];
  for (int i = 1; i < num_inputs; ++i) {
    TfLiteTensor* t = &context->tensors[node->inputs->data[i]];
    TF_LITE_ENSURE_EQ(context, t->dims->size, t0->dims->size);
    TF_LITE_ENSURE_EQ(context, t->type, input_type);
    for (int d = 0; d < t0->dims->size; ++d) {
      if (d == axis) {
        sum_axis += t->dims->data[axis];
      } else {
        TF_LITE_ENSURE_EQ(context, t->dims->data[d], t0->dims->data[d]);
      }
    }
  }

  TfLiteIntArray* output_size = TfLiteIntArrayCreate(t0->dims->size);
  for (int d = 0; d < t0->dims->size; ++d) {
    output_size->data[d] = (d == axis) ? sum_axis : t0->dims->data[d];
  }

  TfLiteTensor* output = &context->tensors[node->outputs->data[0]];
  TF_LITE_ENSURE_EQ(context, output->type, input_type);

  return context->ResizeTensor(context, output, output_size);
}

template <KernelType kernel_type>
TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  auto* params =
      reinterpret_cast<TfLiteConcatenationParams*>(node->builtin_data);
  int axis = params->axis;
  TfLiteTensor* output = &context->tensors[node->outputs->data[0]];
  if (axis < 0) axis += output->dims->size;

// TODO(ahentz): Creating 'all_inputs' below is not very efficient. We should
// allocate and populate these during Prepare().
// TODO(ycling): Activation function parameter is ignored. For now we dont have
// a model with a Concatenation with fused activation function.
#define TF_LITE_CONCATENATION(type, scalar)                                \
  {                                                                        \
    VectorOfTensors<scalar> all_inputs(*context, *node->inputs);           \
    tflite::ConcatenationParams op_params;                                 \
    op_params.axis = axis;                                                 \
    op_params.inputs_count = node->inputs->size;                           \
    type::Concatenation(op_params, all_inputs.shapes(), all_inputs.data(), \
                        GetTensorShape(output),                            \
                        GetTensorData<scalar>(output));                    \
  }


  switch (output->type) {  // Already know in/outtypes are same.
    case kTfLiteFloat32:
      if (kernel_type == kReference) {
        TF_LITE_CONCATENATION(reference_ops, float);
      } else {
        static_assert("Optimized ops for RISCV not implemented yet.");
      }
      break;
    default:
      context->ReportError(context,
                           "Only float32 are currently supported.");
      return kTfLiteError;
  }

#undef TF_LITE_CONCATENATION

  return kTfLiteOk;
}

#undef TF_LITE_MACRO_DISPATCH

}  // namespace concatenation

TfLiteRegistration* Register_CONCATENATION_REF() {
  static TfLiteRegistration r = {
      nullptr, nullptr, concatenation::Prepare,
      concatenation::Eval<concatenation::kReference>};
  return &r;
}

TfLiteRegistration* Register_CONCATENATION_OPT() {
  static TfLiteRegistration r = {
      nullptr, nullptr, concatenation::Prepare,
      concatenation::Eval<concatenation::kOptimized>};
  return &r;
}

TfLiteRegistration* Register_CONCATENATION() {
  // TODO(ahentz): It turns out the two versions of Concatenation are almost
  // identical, so we should consider removing one.
  return Register_CONCATENATION_OPT();
}

}  // namespace riscv
}  // namespace ops
}  // namespace tflite
