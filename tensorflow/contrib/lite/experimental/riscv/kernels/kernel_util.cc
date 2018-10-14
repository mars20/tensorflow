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
#include "tensorflow/contrib/lite/experimental/riscv/kernels/kernel_util.h"

#include <algorithm>
#include <cmath>
#include <memory>

namespace tflite {

bool HaveSameShapes(const TfLiteTensor* input1, const TfLiteTensor* input2) {
  return TfLiteIntArrayEqual(input1->dims, input2->dims);
}

TfLiteStatus CalculateShapeForBroadcast(TfLiteContext* context,
                                        const TfLiteTensor* input1,
                                        const TfLiteTensor* input2,
                                        TfLiteIntArray** output_shape) {
  int64_t dims1 = NumDimensions(input1);
  int64_t dims2 = NumDimensions(input2);
  int64_t out_dims = std::max(dims1, dims2);
  std::unique_ptr<TfLiteIntArray, void (*)(TfLiteIntArray*)> shape(
      TfLiteIntArrayCreate(out_dims), TfLiteIntArrayFree);
  for (int i = 0; i < out_dims; ++i) {
    int64_t d1 = i >= dims1 ? 1 : SizeOfDimension(input1, dims1 - i - 1);
    int64_t d2 = i >= dims2 ? 1 : SizeOfDimension(input2, dims2 - i - 1);
    TF_LITE_ENSURE(context, d1 == d2 || d1 == 1 || d2 == 1);
    shape->data[out_dims - i - 1] = std::max(d1, d2);
  }
  *output_shape = shape.release();
  return kTfLiteOk;
}

}  // namespace tflite
