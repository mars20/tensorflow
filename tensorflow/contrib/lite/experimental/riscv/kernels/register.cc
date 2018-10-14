#include "tensorflow/contrib/lite/experimental/riscv/kernels/register.h"
#include "tensorflow/contrib/lite/util.h"

namespace tflite {
namespace ops {
namespace riscv {

TfLiteRegistration* Register_FULLY_CONNECTED();
TfLiteRegistration* Register_ADD();
// From activations.cc
TfLiteRegistration* Register_RELU();
TfLiteRegistration* Register_RELU_N1_TO_1();
TfLiteRegistration* Register_RELU6();
TfLiteRegistration* Register_TANH();
TfLiteRegistration* Register_LOGISTIC();
TfLiteRegistration* Register_SOFTMAX();
TfLiteRegistration* Register_LOG_SOFTMAX();
TfLiteRegistration* Register_PRELU();

TfLiteRegistration* Register_ARG_MAX();
TfLiteRegistration* Register_ARG_MIN();
TfLiteRegistration* Register_RNN();
TfLiteRegistration* Register_CONV_2D();
TfLiteRegistration* Register_DEPTHWISE_CONV_2D();

TfLiteRegistration* Register_AVERAGE_POOL_2D();
TfLiteRegistration* Register_MAX_POOL_2D();
TfLiteRegistration* Register_L2_POOL_2D();

TfLiteRegistration* Register_RESHAPE();
TfLiteRegistration* Register_SQUEEZE();

const TfLiteRegistration* RiscvOpResolver::FindOp(tflite::BuiltinOperator op,
                                                    int version) const {
  return MutableOpResolver::FindOp(op, version);
}

RiscvOpResolver::RiscvOpResolver() {
  AddBuiltin(BuiltinOperator_FULLY_CONNECTED, Register_FULLY_CONNECTED());
  AddBuiltin(BuiltinOperator_ADD, Register_ADD());
  AddBuiltin(BuiltinOperator_RELU, Register_RELU());
  AddBuiltin(BuiltinOperator_RELU_N1_TO_1, Register_RELU_N1_TO_1());
  AddBuiltin(BuiltinOperator_RELU6, Register_RELU6());
  AddBuiltin(BuiltinOperator_TANH, Register_TANH());
  AddBuiltin(BuiltinOperator_LOGISTIC, Register_LOGISTIC());
  AddBuiltin(BuiltinOperator_SOFTMAX, Register_SOFTMAX());
  AddBuiltin(BuiltinOperator_LOG_SOFTMAX, Register_LOG_SOFTMAX());
  AddBuiltin(BuiltinOperator_PRELU, Register_PRELU());
  AddBuiltin(BuiltinOperator_ARG_MAX, Register_ARG_MAX());
  AddBuiltin(BuiltinOperator_ARG_MIN, Register_ARG_MIN());
  AddBuiltin(BuiltinOperator_RNN, Register_RNN());
  AddBuiltin(BuiltinOperator_CONV_2D, Register_CONV_2D());
  AddBuiltin(BuiltinOperator_DEPTHWISE_CONV_2D, Register_DEPTHWISE_CONV_2D());
  AddBuiltin(BuiltinOperator_AVERAGE_POOL_2D, Register_AVERAGE_POOL_2D());
  AddBuiltin(BuiltinOperator_MAX_POOL_2D, Register_MAX_POOL_2D());
  AddBuiltin(BuiltinOperator_L2_POOL_2D, Register_L2_POOL_2D());
  AddBuiltin(BuiltinOperator_RESHAPE, Register_RESHAPE());
  AddBuiltin(BuiltinOperator_SQUEEZE, Register_SQUEEZE());
}

}  // namespace riscv
}  // namespace ops
}  // namespace tflite