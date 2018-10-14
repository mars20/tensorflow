#ifndef TENSORFLOW_CONTRIB_LITE_EXPERIMENTAL_RISCV_KERNELS_REGISTER_H_
#define TENSORFLOW_CONTRIB_LITE_EXPERIMENTAL_RISCV_KERNELS_REGISTER_H_

#include <unordered_map>
#include "tensorflow/contrib/lite/context.h"
#include "tensorflow/contrib/lite/model.h"

namespace tflite {
namespace ops {
namespace riscv {

class RiscvOpResolver : public MutableOpResolver {
 public:
  RiscvOpResolver();

  const TfLiteRegistration* FindOp(tflite::BuiltinOperator op,
                                   int version) const override;
};

}  // namespace riscv
}  // namespace ops
}  // namespace tflite

#endif  // TENSORFLOW_CONTRIB_LITE_EXPERIMENTAL_RISCV_KERNELS_REGISTER_H_