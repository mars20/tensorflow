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

namespace conv_test {

class BaseConvolutionOpModel : public SingleOpModel {
 public:
  // TODO(ahentz): Also test different activation types, bias, padding types,
  // stride values.
  BaseConvolutionOpModel(
      const TensorData& input, const TensorData& filter,
      const TensorData& output, int stride_width = 2, int stride_height = 2,
      enum Padding padding = Padding_VALID,
      enum ActivationFunctionType activation = ActivationFunctionType_NONE,
      int dilation_width_factor = 1, int dilation_height_factor = 1) {
    input_ = AddInput(input);
    filter_ = AddInput(filter);

    int bias_size = GetShape(filter_)[0];
    if (input.type == TensorType_FLOAT32) {
      bias_ = AddInput({TensorType_FLOAT32, {bias_size}});
    }

    output_ = AddOutput(output);

    SetBuiltinOp(BuiltinOperator_CONV_2D, BuiltinOptions_Conv2DOptions,
                 CreateConv2DOptions(
                     builder_, padding, stride_width, stride_height, activation,
                     dilation_width_factor, dilation_height_factor)
                     .Union());

    // resolver_ = absl::make_unique<SingleOpResolver>(BuiltinOperator_CONV_2D,
    //                                                 registration);
    BuildInterpreter({GetShape(input_), GetShape(filter_), GetShape(bias_)});
  }

 protected:
  int input_;
  int filter_;
  int bias_;
  int output_;
};

class ConvolutionOpModel : public BaseConvolutionOpModel {
 public:
  using BaseConvolutionOpModel::BaseConvolutionOpModel;

  void SetFilter(std::initializer_list<float> f) { PopulateTensor(filter_, f); }

  void SetBias(std::initializer_list<float> f) { PopulateTensor(bias_, f); }

  void SetInput(std::initializer_list<float> data) {
    PopulateTensor(input_, data);
  }
  std::vector<float> GetOutput() { return ExtractVector<float>(output_); }
};

void TestConvSimpleTestFloat32() {
  ConvolutionOpModel m({TensorType_FLOAT32, {2, 2, 4, 1}},
                       {TensorType_FLOAT32, {3, 2, 2, 1}},
                       {TensorType_FLOAT32, {}});

  m.SetInput({
      // First batch
      1, 1, 1, 1,  // row = 1
      2, 2, 2, 2,  // row = 2
      // Second batch
      1, 2, 3, 4,  // row = 1
      1, 2, 3, 4,  // row = 2
  });
  m.SetFilter({
      1, 2, 3, 4,    // first 2x2 filter
      -1, 1, -1, 1,  // second 2x2 filter
      -1, -1, 1, 1,  // third 2x2 filter
  });
  m.SetBias({1, 2, 3});

  m.Invoke();

  std::vector<float> result = m.GetOutput();
  std::vector<float> reference = {
      18, 2, 5,  // first batch, left
      18, 2, 5,  // first batch, right
      17, 4, 3,  // second batch, left
      37, 4, 3,  // second batch, right
  };
  CHECK(isNearlyEqual(result, reference) == true);
}

// This test's output is equivalent to the SimpleTestFloat32
// because we break each input into two channels, each with half of the value,
// while keeping the filters for each channel equivalent.
//
// 2 * (A/2) * B = A * B, where the left side is this new test.
void TestConvSimpleTestFloat32WithChannels() {
  ConvolutionOpModel m({TensorType_FLOAT32, {2, 2, 4, 2}},
                       {TensorType_FLOAT32, {3, 2, 2, 2}},
                       {TensorType_FLOAT32, {}});

  m.SetInput({
      // First batch
      0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,  // row = 1
      1, 1, 1, 1, 1, 1, 1, 1,                  // row = 2
      // Second batch
      0.5, 0.5, 1, 1, 1.5, 1.5, 2, 2,  // row = 1
      0.5, 0.5, 1, 1, 1.5, 1.5, 2, 2   // row = 2
  });
  m.SetFilter({
      1,  1,  2,  2,  3,  3,  4, 4,  // first 2x2 filter
      -1, -1, 1,  1,  -1, -1, 1, 1,  // second 2x2 filter
      -1, -1, -1, -1, 1,  1,  1, 1   // third 2x2 filter
  });
  m.SetBias({1, 2, 3});

  m.Invoke();

  std::vector<float> result = m.GetOutput();
  std::vector<float> reference = {
      18, 2, 5,  // first batch, left
      18, 2, 5,  // first batch, right
      17, 4, 3,  // second batch, left
      37, 4, 3,  // second batch, right
  };
  CHECK(isNearlyEqual(result, reference) == true);
}

void TestConvSimpleTestFloat32WithAnisotropicStrides() {
  ConvolutionOpModel m({TensorType_FLOAT32, {1, 3, 6, 1}},
                       {TensorType_FLOAT32, {1, 2, 2, 1}},
                       {TensorType_FLOAT32, {}},
                       /*stride_width=*/3, /*stride_height=*/1);
  m.SetInput({
      3, 2, 1, -1, -2, -3,  //
      4, 3, 2, -2, -3, -4,  //
      5, 4, 3, -3, -4, -5,  //
  });
  m.SetFilter({
      1, 2,  //
      3, 4,  //
  });
  m.SetBias({-1});
  m.Invoke();

  std::vector<float> result = m.GetOutput();
  std::vector<float> reference = {
      30, -24,  //
      40, -34,  //
  };
  CHECK(isNearlyEqual(result, reference) == true);
}

void TestConvHandCalculatedFloat32() {
  const int depth = 1;
  const int image_width = 4;
  const int image_height = 3;
  const int image_batch_count = 1;
  const int filter_size = 3;
  const int filter_count = 1;
  const int stride_width = 1;
  const int stride_height = 1;
  const Padding padding = Padding_SAME;
  ConvolutionOpModel m(
      {TensorType_FLOAT32,
       {image_batch_count, image_height, image_width, depth}},
      {TensorType_FLOAT32, {depth, filter_size, filter_size, filter_count}},
      {TensorType_FLOAT32, {}}, stride_width, stride_height, padding);

  // The image matrix is:
  // |  1 |  2 |  3 |  4 |
  // |  5 |  6 |  7 |  8 |
  // |  9 | 10 | 11 | 12 |
  m.SetInput({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
  // The filter matrix is:
  // | 1 | 4 | 7 |
  // | 2 | 5 | 8 |
  // | 3 | 6 | 9 |
  m.SetFilter({1, 4, 7, 2, 5, 8, 3, 6, 9});
  // No bias for this test.
  m.SetBias({0});

  m.Invoke();
  // We're sliding the 3x3 filter across the 3x4 image, with accesses outside
  // the input set to zero because we're using the 'SAME' padding mode.
  // The calculations behind the expected output are:
  // (1*0)+(4*0)+(7*0)+(2*0)+(5*1)+(8*2)+(3*0)+(6*5)+(9*6)=105
  // (1*0)+(4*0)+(7*0)+(2*1)+(5*2)+(8*3)+(3*5)+(6*6)+(9*7)=150
  // (1*0)+(4*0)+(7*0)+(2*2)+(5*3)+(8*4)+(3*6)+(6*7)+(9*8)=183
  // (1*0)+(4*0)+(7*0)+(2*3)+(5*4)+(8*0)+(3*7)+(6*8)+(9*0)=95
  // (1*0)+(4*1)+(7*2)+(2*0)+(5*5)+(8*6)+(3*0)+(6*9)+(9*10)=235
  // (1*1)+(4*2)+(7*3)+(2*5)+(5*6)+(8*7)+(3*9)+(6*10)+(9*11)=312
  // (1*2)+(4*3)+(7*4)+(2*6)+(5*7)+(8*8)+(3*10)+(6*11)+(9*12)=357
  // (1*3)+(4*4)+(7*0)+(2*7)+(5*8)+(8*0)+(3*11)+(6*12)+(9*0)=178
  // (1*0)+(4*5)+(7*6)+(2*0)+(5*9)+(8*10)+(3*0)+(6*0)+(9*0)=187
  // (1*5)+(4*6)+(7*7)+(2*9)+(5*10)+(8*11)+(3*0)+(6*0)+(9*0)=234
  // (1*6)+(4*7)+(7*8)+(2*10)+(5*11)+(8*12)+(3*0)+(6*0)+(9*0)=261
  // (1*7)+(4*11)+(7*0)+(2*8)+(5*12)+(8*0)+(3*0)+(6*0)+(9*0)=121
  // This means we should end up with this matrix:
  // |  105  |  150  |  183  |   95  |
  // |  235  |  312  |  357  |  178  |
  // |  187  |  234  |  261  |  121  |

  std::vector<float> result = m.GetOutput();
  std::vector<float> reference = {105, 150, 183, 95,  235, 312,
                                  357, 178, 187, 234, 261, 121};
  CHECK(isNearlyEqual(result, reference) == true);
}

void TestConvHandCalculatedWithBiasFloat32() {
  const int depth = 1;
  const int image_width = 4;
  const int image_height = 3;
  const int image_batch_count = 1;
  const int filter_size = 3;
  const int filter_count = 1;
  const int stride_width = 1;
  const int stride_height = 1;
  const Padding padding = Padding_SAME;
  ConvolutionOpModel m(
      {TensorType_FLOAT32,
       {image_batch_count, image_height, image_width, depth}},
      {TensorType_FLOAT32, {depth, filter_size, filter_size, filter_count}},
      {TensorType_FLOAT32, {}}, stride_width, stride_height, padding);

  // The image matrix is:
  // |  1 |  2 |  3 |  4 |
  // |  5 |  6 |  7 |  8 |
  // |  9 | 10 | 11 | 12 |
  m.SetInput({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
  // The filter matrix is:
  // | 1 | 4 | 7 |
  // | 2 | 5 | 8 |
  // | 3 | 6 | 9 |
  m.SetFilter({1, 4, 7, 2, 5, 8, 3, 6, 9});
  // Bias is | 10 |.
  m.SetBias({10});

  m.Invoke();
  // We're sliding the 3x3 filter across the 3x4 image, with accesses outside
  // the input set to zero because we're using the 'SAME' padding mode.
  // The calculations behind the expected output are:
  // (1*0)+(4*0)+(7*0)+(2*0)+(5*1)+(8*2)+(3*0)+(6*5)+(9*6)+10=115
  // (1*0)+(4*0)+(7*0)+(2*1)+(5*2)+(8*3)+(3*5)+(6*6)+(9*7)+10=160
  // (1*0)+(4*0)+(7*0)+(2*2)+(5*3)+(8*4)+(3*6)+(6*7)+(9*8)+10=193
  // (1*0)+(4*0)+(7*0)+(2*3)+(5*4)+(8*0)+(3*7)+(6*8)+(9*0)+10=105
  // (1*0)+(4*1)+(7*2)+(2*0)+(5*5)+(8*6)+(3*0)+(6*9)+(9*10)+10=245
  // (1*1)+(4*2)+(7*3)+(2*5)+(5*6)+(8*7)+(3*9)+(6*10)+(9*11)+10=322
  // (1*2)+(4*3)+(7*4)+(2*6)+(5*7)+(8*8)+(3*10)+(6*11)+(9*12)+10=367
  // (1*3)+(4*4)+(7*0)+(2*7)+(5*8)+(8*0)+(3*11)+(6*12)+(9*0)+10=188
  // (1*0)+(4*5)+(7*6)+(2*0)+(5*9)+(8*10)+(3*0)+(6*0)+(9*0)+10=197
  // (1*5)+(4*6)+(7*7)+(2*9)+(5*10)+(8*11)+(3*0)+(6*0)+(9*0)+10=244
  // (1*6)+(4*7)+(7*8)+(2*10)+(5*11)+(8*12)+(3*0)+(6*0)+(9*0)+10=271
  // (1*7)+(4*11)+(7*0)+(2*8)+(5*12)+(8*0)+(3*0)+(6*0)+(9*0)+10=131
  // This means we should end up with this matrix:
  // |  115  |  160  |  193  |  105  |
  // |  245  |  322  |  367  |  188  |
  // |  197  |  244  |  271  |  131  |

  std::vector<float> result = m.GetOutput();
  std::vector<float> reference = {115, 160, 193, 105, 245, 322,
                                  367, 188, 197, 244, 271, 131};
  CHECK(isNearlyEqual(result, reference) == true);
}

void TestConvHandCalculatedWithReluFloat32() {
  const int depth = 1;
  const int image_width = 4;
  const int image_height = 3;
  const int image_batch_count = 1;
  const int filter_size = 3;
  const int filter_count = 1;
  const int stride_width = 1;
  const int stride_height = 1;
  const Padding padding = Padding_SAME;
  ConvolutionOpModel m(
      {TensorType_FLOAT32,
       {image_batch_count, image_height, image_width, depth}},
      {TensorType_FLOAT32, {depth, filter_size, filter_size, filter_count}},
      {TensorType_FLOAT32, {}}, stride_width, stride_height, padding,
      ActivationFunctionType_RELU);

  // The image matrix is:
  // |  1 |  2 |  3 |  4 |
  // |  5 |  6 |  7 |  8 |
  // |  9 | 10 | 11 | 12 |
  m.SetInput({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
  // The filter matrix is:
  // | 1 | 4 | 7 |
  // | 2 | 5 | 8 |
  // | 3 | 6 | 9 |
  m.SetFilter({1, 4, 7, 2, 5, 8, 3, 6, 9});
  // Bias is | -200 |.
  m.SetBias({-200});

  m.Invoke();
  // We're sliding the 3x3 filter across the 3x4 image, with accesses outside
  // the input set to zero because we're using the 'SAME' padding mode.
  // The calculations behind the expected output are:
  // (1*0)+(4*0)+(7*0)+(2*0)+(5*1)+(8*2)+(3*0)+(6*5)+(9*6)-200=-95
  // (1*0)+(4*0)+(7*0)+(2*1)+(5*2)+(8*3)+(3*5)+(6*6)+(9*7)-200=-50
  // (1*0)+(4*0)+(7*0)+(2*2)+(5*3)+(8*4)+(3*6)+(6*7)+(9*8)-200=-17
  // (1*0)+(4*0)+(7*0)+(2*3)+(5*4)+(8*0)+(3*7)+(6*8)+(9*0)-200=-105
  // (1*0)+(4*1)+(7*2)+(2*0)+(5*5)+(8*6)+(3*0)+(6*9)+(9*10)-200=35
  // (1*1)+(4*2)+(7*3)+(2*5)+(5*6)+(8*7)+(3*9)+(6*10)+(9*11)-200=112
  // (1*2)+(4*3)+(7*4)+(2*6)+(5*7)+(8*8)+(3*10)+(6*11)+(9*12)-200=157
  // (1*3)+(4*4)+(7*0)+(2*7)+(5*8)+(8*0)+(3*11)+(6*12)+(9*0)-200=-22
  // (1*0)+(4*5)+(7*6)+(2*0)+(5*9)+(8*10)+(3*0)+(6*0)+(9*0)-200=-13
  // (1*5)+(4*6)+(7*7)+(2*9)+(5*10)+(8*11)+(3*0)+(6*0)+(9*0)-200=34
  // (1*6)+(4*7)+(7*8)+(2*10)+(5*11)+(8*12)+(3*0)+(6*0)+(9*0)-200=61
  // (1*7)+(4*11)+(7*0)+(2*8)+(5*12)+(8*0)+(3*0)+(6*0)+(9*0)-200=-79
  // All negative values are gated to zero by the Relu activation function.
  // This means we should end up with this matrix:
  // |   0 |   0 |   0 |   0 |
  // |  35 | 112 | 157 |   0 |
  // |   0 |  34 |  61 |   0 |
  std::vector<float> result = m.GetOutput();
  std::vector<float> reference = {0, 0, 0, 0, 35, 112, 157, 0, 0, 34, 61, 0};
  CHECK(isNearlyEqual(result, reference) == true);
}

void TestConvHandCalculatedValidFloat32() {
  const int depth = 1;
  const int image_width = 4;
  const int image_height = 3;
  const int image_batch_count = 1;
  const int filter_size = 3;
  const int filter_count = 1;
  const int stride_width = 1;
  const int stride_height = 1;
  const Padding padding = Padding_VALID;
  ConvolutionOpModel m(
      {TensorType_FLOAT32,
       {image_batch_count, image_height, image_width, depth}},
      {TensorType_FLOAT32, {depth, filter_size, filter_size, filter_count}},
      {TensorType_FLOAT32, {}}, stride_width, stride_height, padding);

  // The image matrix is:
  // |  1 |  2 |  3 |  4 |
  // |  5 |  6 |  7 |  8 |
  // |  9 | 10 | 11 | 12 |
  m.SetInput({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
  // The filter matrix is:
  // | 1 | 4 | 7 |
  // | 2 | 5 | 8 |
  // | 3 | 6 | 9 |
  m.SetFilter({1, 4, 7, 2, 5, 8, 3, 6, 9});
  // No bias for this test.
  m.SetBias({0});

  m.Invoke();
  // We're sliding the 3x3 filter across the 3x4 image, with no accesses outside
  // the input because we're using the 'VALID' padding mode, giving a 2x1
  // output.
  // The calculations behind the expected output are:
  // (1*1)+(4*2)+(7*3)+(2*5)+(5*6)+(8*7)+(3*9)+(6*10)+(9*11)=312
  // (1*2)+(4*3)+(7*4)+(2*6)+(5*7)+(8*8)+(3*10)+(6*11)+(9*12)=357
  // This means we should end up with this matrix:
  // |  312  |  357  |
  std::vector<float> result = m.GetOutput();
  std::vector<float> reference = {312, 357};
  CHECK(isNearlyEqual(result, reference) == true);
}

void TestConvSimpleTestFloatWithDilation() {
  const int depth = 1;
  const int image_width = 9;
  const int image_height = 9;
  const int image_batch_count = 1;
  const int filter_size = 3;
  const int filter_count = 1;
  const int stride_width = 1;
  const int stride_height = 1;
  const int dilation_width_factor = 3;
  const int dilation_height_factor = 3;
  const Padding padding = Padding_VALID;
  ConvolutionOpModel m(
      {TensorType_FLOAT32,
       {image_batch_count, image_height, image_width, depth}},
      {TensorType_FLOAT32, {depth, filter_size, filter_size, filter_count}},
      {TensorType_FLOAT32, {}}, stride_width, stride_height, padding,
      ActivationFunctionType_NONE, dilation_width_factor,
      dilation_height_factor);

  // The image matrix is:
  // | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
  // | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
  // | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
  // | 0 | 0 | 0 | 1 | 1 | 1 | 0 | 0 | 0 |
  // | 0 | 0 | 0 | 1 | 1 | 1 | 0 | 0 | 0 |
  // | 0 | 0 | 0 | 1 | 1 | 1 | 0 | 0 | 0 |
  // | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
  // | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
  // | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
  // clang-format off
  m.SetInput({0, 0, 0, 0, 0, 0, 0, 0, 0,
              0, 0, 0, 0, 0, 0, 0, 0, 0,
              0, 0, 0, 0, 0, 0, 0, 0, 0,
              0, 0, 0, 1, 1, 1, 0, 0, 0,
              0, 0, 0, 1, 1, 1, 0, 0, 0,
              0, 0, 0, 1, 1, 1, 0, 0, 0,
              0, 0, 0, 0, 0, 0, 0, 0, 0,
              0, 0, 0, 0, 0, 0, 0, 0, 0,
              0, 0, 0, 0, 0, 0, 0, 0, 0});
  // clang-format on
  // The filter matrix is:
  // | 1 | 2 | 3 |
  // | 4 | 5 | 6 |
  // | 7 | 8 | 9 |
  m.SetFilter({1, 2, 3, 4, 5, 6, 7, 8, 9});
  // No bias for this test.
  m.SetBias({0});
  m.Invoke();

  // Since the dilation rate is 3 this will reduce the size of the output from
  // 10x10 to 3x3 of all 5s. Specifically:
  // | 5 | 5 | 5 |
  // | 5 | 5 | 5 |
  // | 5 | 5 | 5 |
  std::vector<float> result = m.GetOutput();
  std::vector<float> reference = {5, 5, 5, 5, 5, 5, 5, 5, 5};
  CHECK(isNearlyEqual(result, reference) == true);
}
}  // namespace conv_test
}  // namespace tflite

int main(int argc, char** argv) {
  tflite::conv_test::TestConvSimpleTestFloat32();
  tflite::conv_test::TestConvSimpleTestFloat32WithChannels();
  tflite::conv_test::TestConvSimpleTestFloat32WithAnisotropicStrides();
  tflite::conv_test::TestConvHandCalculatedFloat32();
  tflite::conv_test::TestConvHandCalculatedWithBiasFloat32();
  tflite::conv_test::TestConvHandCalculatedWithReluFloat32();
  tflite::conv_test::TestConvHandCalculatedValidFloat32();
  tflite::conv_test::TestConvSimpleTestFloatWithDilation();
}
