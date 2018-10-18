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
#include <cstdarg>
#include "tensorflow/contrib/lite/experimental/riscv/kernels/register.h"
#include "tensorflow/contrib/lite/experimental/riscv/kernels/test_util.h"
#include "tensorflow/contrib/lite/interpreter.h"
#include "tensorflow/contrib/lite/model.h"

namespace tflite {
namespace pooling_test {

class BasePoolingOpModel : public SingleOpModel {
 public:
  // TODO(ahentz): Also test different activation types, bias, padding types,
  // stride values.
  BasePoolingOpModel(BuiltinOperator type, const TensorData& input,
                     int filter_width, int filter_height,
                     const TensorData& output) {
    input_ = AddInput(input);
    output_ = AddOutput(output);

    SetBuiltinOp(
        type, BuiltinOptions_Pool2DOptions,
        CreatePool2DOptions(builder_, Padding_VALID, 2, 2, filter_width,
                            filter_height, ActivationFunctionType_NONE)
            .Union());

    BuildInterpreter({GetShape(input_)});
  }

 protected:
  int input_;
  int output_;
};

class FloatPoolingOpModel : public BasePoolingOpModel {
 public:
  using BasePoolingOpModel::BasePoolingOpModel;

  void SetInput(std::initializer_list<float> data) {
    PopulateTensor(input_, data);
  }

  std::vector<float> GetOutput() { return ExtractVector<float>(output_); }
};

void TestFloatPoolingAveragePool() {
  FloatPoolingOpModel m(BuiltinOperator_AVERAGE_POOL_2D,
                        /*input=*/{TensorType_FLOAT32, {1, 2, 4, 1}},
                        /*filter_width=*/2, /*filter_height=*/2,
                        /*output=*/{TensorType_FLOAT32, {}});
  m.SetInput({
      0, 6, 2, 4,   //
      3, 2, 10, 7,  //
  });
  m.Invoke();
  std::vector<float> result = m.GetOutput();
  std::vector<float> reference = {2.75, 5.75};
  CHECK(isNearlyEqual(result, reference) == true);
}

void TestFloatPoolingMaxPool() {
  FloatPoolingOpModel m(BuiltinOperator_MAX_POOL_2D,
                        /*input=*/{TensorType_FLOAT32, {1, 2, 4, 1}},
                        /*filter_width=*/2, /*filter_height=*/2,
                        /*output=*/{TensorType_FLOAT32, {}});
  m.SetInput({
      0, 6, 2, 4,   //
      3, 2, 10, 7,  //
  });
  m.Invoke();

  std::vector<float> result = m.GetOutput();
  std::vector<float> reference = {6, 10};
  CHECK(isNearlyEqual(result, reference) == true);
}

void TestFloatPoolingL2Pool() {
  FloatPoolingOpModel m(BuiltinOperator_L2_POOL_2D,
                        /*input=*/{TensorType_FLOAT32, {1, 2, 4, 1}},
                        /*filter_width=*/2, /*filter_height=*/2,
                        /*output=*/{TensorType_FLOAT32, {}});
  m.SetInput({
      0, 6, 2, 4,   //
      3, 2, 10, 7,  //
  });
  m.Invoke();
  std::vector<float> result = m.GetOutput();
  std::vector<float> reference = {3.5, 6.5};
  CHECK(isNearlyEqual(result, reference) == true);
}

}  // namespace pooling_test
}  // namespace tflite

int main(int argc, char** argv) {
  tflite::pooling_test::TestFloatPoolingAveragePool();
  tflite::pooling_test::TestFloatPoolingMaxPool();
  tflite::pooling_test::TestFloatPoolingL2Pool();
}
