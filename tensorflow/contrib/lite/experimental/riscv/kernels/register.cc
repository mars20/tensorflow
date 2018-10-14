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
#include "tensorflow/contrib/lite/experimental/riscv/kernels/register.h"
#include "tensorflow/contrib/lite/util.h"

namespace tflite {
namespace ops {
namespace riscv {

TfLiteRegistration* Register_FULLY_CONNECTED();
TfLiteRegistration* Register_ADD();

const TfLiteRegistration* RiscvOpResolver::FindOp(tflite::BuiltinOperator op,
                                                    int version) const {
  return MutableOpResolver::FindOp(op, version);
}

RiscvOpResolver::RiscvOpResolver() {
  AddBuiltin(BuiltinOperator_FULLY_CONNECTED, Register_FULLY_CONNECTED());
  AddBuiltin(BuiltinOperator_ADD, Register_ADD());
}

}  // namespace riscv
}  // namespace ops
}  // namespace tflite