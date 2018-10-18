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
#include "tensorflow/contrib/lite/experimental/riscv/kernels/test_util.h"
#include "tensorflow/contrib/lite/interpreter.h"
#include "tensorflow/contrib/lite/model.h"

namespace tflite {
namespace add_test {

class BaseAddOpModel : public SingleOpModel {
 public:
  BaseAddOpModel(const TensorData& input1, const TensorData& input2,
                 const TensorData& output,
                 ActivationFunctionType activation_type) {
    input1_ = AddInput(input1);
    input2_ = AddInput(input2);
    output_ = AddOutput(output);
    SetBuiltinOp(BuiltinOperator_ADD, BuiltinOptions_AddOptions,
                 CreateAddOptions(builder_, activation_type).Union());
    BuildInterpreter({GetShape(input1_), GetShape(input2_)});
  }

  int input1() { return input1_; }
  int input2() { return input2_; }

 protected:
  int input1_;
  int input2_;
  int output_;
};

class FloatAddOpModel : public BaseAddOpModel {
 public:
  using BaseAddOpModel::BaseAddOpModel;

  std::vector<float> GetOutput() { return ExtractVector<float>(output_); }
};

void TestAddNoActivation() {
  FloatAddOpModel m({TensorType_FLOAT32, {1, 2, 2, 1}},
                    {TensorType_FLOAT32, {1, 2, 2, 1}},
                    {TensorType_FLOAT32, {}}, ActivationFunctionType_NONE);
  m.PopulateTensor<float>(m.input1(), {-2.0, 0.2, 0.7, 0.8});
  m.PopulateTensor<float>(m.input2(), {0.1, 0.2, 0.3, 0.5});
  m.Invoke();
  std::vector<float> result = m.GetOutput();
  std::vector<float> reference = {-1.9, 0.4, 1.0, 1.3};
  CHECK(isNearlyEqual(result, reference) == true);
}

void TestAddActivationRELU_N1_TO_1() {
  FloatAddOpModel m(
      {TensorType_FLOAT32, {1, 2, 2, 1}}, {TensorType_FLOAT32, {1, 2, 2, 1}},
      {TensorType_FLOAT32, {}}, ActivationFunctionType_RELU_N1_TO_1);
  m.PopulateTensor<float>(m.input1(), {-2.0, 0.2, 0.7, 0.8});
  m.PopulateTensor<float>(m.input2(), {0.1, 0.2, 0.3, 0.5});
  m.Invoke();
  std::vector<float> result = m.GetOutput();
  std::vector<float> reference = {-1.0, 0.4, 1.0, 1.0};
  CHECK(isNearlyEqual(result, reference) == true);
}

void TestAddVariousInputShapes() {
  std::vector<std::initializer_list<int>> test_shapes = {
      {6}, {2, 3}, {2, 1, 3}, {1, 3, 1, 2}};
  for (int i = 0; i < test_shapes.size(); ++i) {
    FloatAddOpModel m({TensorType_FLOAT32, test_shapes[i]},
                      {TensorType_FLOAT32, test_shapes[i]},
                      {TensorType_FLOAT32, {}}, ActivationFunctionType_NONE);
    m.PopulateTensor<float>(m.input1(), {-2.0, 0.2, 0.7, 0.8, 1.1, 2.0});
    m.PopulateTensor<float>(m.input2(), {0.1, 0.2, 0.3, 0.5, 1.1, 0.1});
    m.Invoke();
    std::vector<float> result = m.GetOutput();
    std::vector<float> reference = {-1.9, 0.4, 1.0, 1.3, 2.2, 2.1};
    CHECK(isNearlyEqual(result, reference) == true);
    printf("With shape number %d\n", i);
  }
}

}  // namespace add_test
}  // namespace tflite
int main(int argc, char** argv) {
  tflite::add_test::TestAddNoActivation();
  tflite::add_test::TestAddActivationRELU_N1_TO_1();
  tflite::add_test::TestAddVariousInputShapes();
}
