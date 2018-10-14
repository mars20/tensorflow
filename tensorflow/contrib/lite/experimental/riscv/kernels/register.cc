#include "tensorflow/contrib/lite/experimental/riscv/kernels/register.h"
#include "tensorflow/contrib/lite/util.h"

namespace tflite {
namespace ops {
namespace riscv {

const TfLiteRegistration* RiscvOpResolver::FindOp(tflite::BuiltinOperator op,
                                                    int version) const {
  return MutableOpResolver::FindOp(op, version);
}

RiscvOpResolver::RiscvOpResolver() {

}

}  // namespace riscv
}  // namespace ops
}  // namespace tflite