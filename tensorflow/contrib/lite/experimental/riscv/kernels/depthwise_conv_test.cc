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

#include "tensorflow/contrib/lite/experimental/riscv/kernels/optimized/depthwiseconv_float.h"
#include "tensorflow/contrib/lite/experimental/riscv/kernels/reference/depthwiseconv_float.h"
#include "tensorflow/contrib/lite/kernels/internal/test_util.h"

namespace tflite {
namespace depthwiseconv_test {

class BaseDepthwiseConvolutionOpModel : public SingleOpModel {
 public:
  // TODO(ahentz): Also test different activation types, bias, padding types,
  // stride values.
  BaseDepthwiseConvolutionOpModel(const TensorData& input,
                                  const TensorData& filter,
                                  const TensorData& output,
                                  Padding padding_type,
                                  int dilation_factor = 1) {

    input_ = AddInput(input);
    filter_ = AddInput(filter);

    int bias_size = GetShape(filter_)[3];
    if (input.type == TensorType_FLOAT32) {
      bias_ = AddInput({TensorType_FLOAT32, {bias_size}});
    }

    output_ = AddOutput(output);

    int input_depth = GetShape(input_)[3];
    int output_depth = GetShape(filter_)[3];
    int depth_mul = output_depth / input_depth;

    SetBuiltinOp(
        BuiltinOperator_DEPTHWISE_CONV_2D,
        BuiltinOptions_DepthwiseConv2DOptions,
        CreateDepthwiseConv2DOptions(builder_, padding_type, 1, 1, depth_mul,
                                     ActivationFunctionType_NONE,
                                     dilation_factor, dilation_factor)
            .Union());

    BuildInterpreter({GetShape(input_), GetShape(filter_), GetShape(bias_)});
  }

 protected:
  int input_;
  int filter_;
  int bias_;
  int output_;
};

class DepthwiseConvolutionOpModel : public BaseDepthwiseConvolutionOpModel {
 public:
  using BaseDepthwiseConvolutionOpModel::BaseDepthwiseConvolutionOpModel;

  void SetFilter(std::initializer_list<float> f) { PopulateTensor(filter_, f); }

  void SetBias(std::initializer_list<float> f) { PopulateTensor(bias_, f); }

  void SetInput(std::initializer_list<float> data) {
    PopulateTensor(input_, data);
  }

  std::vector<float> GetOutput() { return ExtractVector<float>(output_); }
};

void TestDepthwiseConvSimpleTest() {
  DepthwiseConvolutionOpModel m({TensorType_FLOAT32, {1, 3, 2, 2}},
                                {TensorType_FLOAT32, {1, 2, 2, 4}},
                                {TensorType_FLOAT32, {}}, Padding_VALID);

  m.SetInput({
      1, 2, 7, 8,    // column 1
      3, 4, 9, 10,   // column 2
      5, 6, 11, 12,  // column 3
  });
  m.SetFilter({
      1, 2, 3, 4,        //
      -9, 10, -11, 12,   //
      5, 6, 7, 8,        //
      13, -14, 15, -16,  //
  });
  m.SetBias({1, 2, 3, 4});

  m.Invoke();
  std::vector<float> result = m.GetOutput();
  std::vector<float> reference = {
      71, -34, 99,  -20,  //
      91, -26, 127, -4,   //
  };
  CHECK(isNearlyEqual(result, reference) == true);
}

void TestDepthwiseConvSimpleDilatedTestPaddingValid() {
  const int depth = 1;
  const int image_width = 9;
  const int image_height = 9;
  const int image_batch_count = 1;
  const int filter_size = 3;
  const int filter_count = 1;
  const int dilation_factor = 3;
  DepthwiseConvolutionOpModel m(
      {TensorType_FLOAT32,
       {image_batch_count, image_height, image_width, depth}},
      {TensorType_FLOAT32, {depth, filter_size, filter_size, filter_count}},
      {TensorType_FLOAT32, {}}, Padding_VALID, dilation_factor);

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

  // Since the dilation rate is 3 this will reduce the size of the output
  // 10x10 to 3x3 of all 5s. Specifically:
  // | 5 | 5 | 5 |
  // | 5 | 5 | 5 |
  // | 5 | 5 | 5 |
  std::vector<float> result = m.GetOutput();
  std::vector<float> reference ={5, 5, 5, 5, 5, 5, 5, 5, 5};
  CHECK(isNearlyEqual(result, reference) == true);
}

void TestDepthwiseConvSimpleDilatedTestPaddingSame() {
  const int depth = 1;
  const int image_width = 3;
  const int image_height = 3;
  const int image_batch_count = 1;
  const int filter_size = 2;
  const int filter_count = 1;
  const int dilation_factor = 2;
  DepthwiseConvolutionOpModel m(
      {TensorType_FLOAT32,
       {image_batch_count, image_height, image_width, depth}},
      {TensorType_FLOAT32, {depth, filter_size, filter_size, filter_count}},
      {TensorType_FLOAT32, {}}, Padding_SAME, dilation_factor);

  // The image matrix is:
  // | 1 | 1 | 1 |
  // | 1 | 1 | 1 |
  // | 1 | 1 | 1 |
  m.SetInput({1, 1, 1, 1, 1, 1, 1, 1, 1});
  // The filter matrix is:
  // | 1 | 2 |
  // | 3 | 4 |
  m.SetFilter({1, 2, 3, 4});
  // No bias for this test.
  m.SetBias({0});
  m.Invoke();

  // Output:
  // | 4 | 7 | 3 |
  // | 6 |10 | 4 |
  // | 2 | 3 | 1 |
  std::vector<float> result = m.GetOutput();
  std::vector<float> reference ={4, 7, 3, 6, 10, 4, 2, 3, 1};
  CHECK(isNearlyEqual(result, reference) == true);
}



// Runs the DepthwiseConv and compares against the reference implementation.
#ifdef RISCV
void TestOneDepthwiseConv(
    const DepthwiseParams& params, const RuntimeShape& input_shape,
    const float* input_data, const RuntimeShape& filter_shape,
    const float* filter_data, const RuntimeShape& bias_shape,
    const float* bias_data, const RuntimeShape& output_shape) {
  const int output_buffer_size = output_shape.FlatSize();
  std::vector<float> output_data(output_buffer_size);
  std::vector<float> reference_output_data(output_buffer_size);
  reference_ops::DepthwiseConv(params, input_shape, input_data, filter_shape,
                               filter_data, bias_shape, bias_data, output_shape,
                               reference_output_data.data());
  optimized_ops::DepthwiseConv(params, input_shape, input_data, filter_shape,
                               filter_data, bias_shape, bias_data, output_shape,
                               output_data.data());

  //isNearlyEqual(output_data, reference_output_data);

  double sum_abs_diff = 0;
  float max_abs_val = 0;
  for (int i = 0; i < output_buffer_size; i++) {
    sum_abs_diff += std::abs(output_data[i] - reference_output_data[i]);
    max_abs_val = std::max(max_abs_val, std::abs(reference_output_data[i]));
  }
  if (sum_abs_diff != 0.f) {
    const float mean_diff =
        static_cast<float>(sum_abs_diff / output_buffer_size);
    const float relative_error = std::abs(mean_diff) / max_abs_val;
    if(relative_error > 1e-5f){
      printf("batch:%d\n input_depth:%d\n input_width:%d\n input_height:%d\n filter_width:%d\n, filter_height:%d\n, depth_multiplier:%d\n, stride:%d\n output_depth:%d\n dilation_width_factor:%d\n dilation_height_factor:%d\n output_activation_min:%d\n output_activation_max:%d\n",input_shape.Dims(0), input_shape.Dims(3), input_shape.Dims(2), input_shape.Dims(1),filter_shape.Dims(2), filter_shape.Dims(1), params.depth_multiplier, params.stride_width, output_shape.Dims(3) , params.dilation_width_factor, params.dilation_height_factor, params.float_activation_min, params.float_activation_max);
    }
    printf("Relative error %f, ref %f\n", relative_error, 1e-5f);
    //ASSERT_LT(relative_error, 1e-5f);
  }
}

// This function picks some random DepthwiseConv params, which may or may not
// be legal. If they're not legal, it returns false. If they're legal,
// it runs the DepthwiseConv test and returns true. This allows the caller
// to loop until a test has been run.
bool TryTestOneDepthwiseConv() {
  // We have to pick a lot of positive values, where we are particularly
  // interested in small values because they are most likely to be special
  // cases in optimized implementations, and secondarily because they allow
  // tests to run fast, which means we can run more tests and get more
  // coverage.
  const int batch = ExponentialRandomPositiveInt(0.9f, 3, 20);
  const int input_depth = ExponentialRandomPositiveInt(0.9f, 6, 50);
  const int input_width = ExponentialRandomPositiveInt(0.9f, 20, 200);
  const int input_height = ExponentialRandomPositiveInt(0.9f, 20, 200);
  const int filter_width = ExponentialRandomPositiveInt(0.9f, 4, 10);
  const int filter_height = ExponentialRandomPositiveInt(0.9f, 4, 10);
  const int depth_multiplier = ExponentialRandomPositiveInt(0.8f, 6, 50);
  const int stride = ExponentialRandomPositiveInt(0.9f, 3, 8);
  const int output_depth = input_depth * depth_multiplier;
  const int dilation_width_factor = RandomElement(std::vector<int>({1, 2, 4}));
  const int dilation_height_factor = RandomElement(std::vector<int>({1, 2, 4}));
  float output_activation_min, output_activation_max;
  FusedActivationFunctionType ac =
      RandomElement(std::vector<FusedActivationFunctionType>(
          {FusedActivationFunctionType::kNone,
           FusedActivationFunctionType::kRelu,
           FusedActivationFunctionType::kRelu1,
           FusedActivationFunctionType::kRelu6}));
  GetActivationMinMax(ac, &output_activation_min, &output_activation_max);
  // The optimized DepthwiseConv implementation currently uses a fixed-size
  // accumulator buffer on the stack, with that size. This currently means
  // that it does not support larger output depths. It CHECK's for it,
  // so it's safe in the sense that if a larger output depth was encountered,
  // it would explicitly fail. We just need to adjust our testing to that
  // constraint.

  const int kMaxSupportedOutputDepth = 1024;
  if (output_depth > kMaxSupportedOutputDepth) {
    return false;
  }

  RuntimeShape input_shape_inference(
      {batch, input_height, input_width, input_depth});
  RuntimeShape output_shape_inference;
  int pad_width, pad_height;
  const auto padding_type =
      UniformRandomInt(0, 1) ? PaddingType::kSame : PaddingType::kValid;
  if (!ComputeConvSizes(input_shape_inference, output_depth, filter_width,
                        filter_height, stride, dilation_width_factor,
                        dilation_height_factor, padding_type,
                        &output_shape_inference, &pad_width, &pad_height)) {
    return false;
  }
  RuntimeShape filter_shape_inference(
      {1, filter_height, filter_width, output_depth});
  RuntimeShape bias_shape_inference({1, 1, 1, output_depth});
  const int input_buffer_size = input_shape_inference.FlatSize();
  const int filter_buffer_size = filter_shape_inference.FlatSize();
  std::vector<float> input_data(input_buffer_size);
  std::vector<float> filter_data(filter_buffer_size);
  std::vector<float> bias_data(output_depth);
  const float input_amplitude = 1.f;
  const float filter_amplitude = 1.f;
  const float bias_amplitude =
      filter_width * filter_height * input_amplitude * filter_amplitude;
  FillRandom(&input_data, -input_amplitude, input_amplitude);
  FillRandom(&filter_data, -filter_amplitude, filter_amplitude);
  FillRandom(&bias_data, -bias_amplitude, bias_amplitude);
  DepthwiseParams op_params;
  op_params.padding_type = PaddingType::kSame;
  op_params.padding_values.width = pad_width;
  op_params.padding_values.height = pad_height;
  op_params.stride_width = stride;
  op_params.stride_height = stride;
  op_params.dilation_width_factor = dilation_width_factor;
  op_params.dilation_height_factor = dilation_height_factor;
  op_params.depth_multiplier = depth_multiplier;
  op_params.float_activation_min = output_activation_min;
  op_params.float_activation_max = output_activation_max;
  TestOneDepthwiseConv(op_params, input_shape_inference, input_data.data(),
                       filter_shape_inference, filter_data.data(),
                       bias_shape_inference, bias_data.data(),
                       output_shape_inference);
  return true;
}

void TestOneDepthwiseConv() {
  while (!TryTestOneDepthwiseConv()) {
  }
}
#endif
}  // namespace depthwiseconv_test
}  // namespace tflite
int main(int argc, char** argv) {
  tflite::depthwiseconv_test::TestDepthwiseConvSimpleTest();
  tflite::depthwiseconv_test::TestDepthwiseConvSimpleDilatedTestPaddingValid();
  tflite::depthwiseconv_test::TestDepthwiseConvSimpleDilatedTestPaddingSame();
  #ifdef RISCV
  const int kTestsToRun = 100;
  for (int i = 0; i < kTestsToRun; i++) {
    tflite::depthwiseconv_test::TestOneDepthwiseConv();
  }
  #endif
}
