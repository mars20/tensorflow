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
#include <cstdlib>

#include "tensorflow/contrib/lite/experimental/riscv/kernels/register.h"
#include "tensorflow/contrib/lite/experimental/riscv/kernels/test_util.h"
#include "tensorflow/contrib/lite/interpreter.h"
#include "tensorflow/contrib/lite/model.h"
#include "tensorflow/contrib/lite/experimental/riscv/profiling/stats.h"

namespace tflite {

namespace benchmark_conv{

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

  void SetFilter(std::vector<int> dim) {
    PopulateTensorRandom<float>(filter_, dim,
                                []() { return static_cast<float>(rand()) / RAND_MAX - 0.5f; });
  }

  void SetBias(std::vector<int> dim) {
    PopulateTensorRandom<float>(bias_, dim,
                                []() { return static_cast<float>(rand()) / RAND_MAX - 0.5f; });
  }

  void SetInput(std::vector<int> dim) {
    PopulateTensorRandom<float>(input_, dim,
                                []() { return static_cast<float>(rand()) / RAND_MAX - 0.5f;});
  }
  std::vector<float> GetOutput() { return ExtractVector<float>(output_); }
};

void ConvBenchmarkFloat32InputWidthHeight(int matrix_size, int num_runs) {
  const int depth = 8;
  const int image_width = matrix_size;
  const int image_height = matrix_size;
  const int image_batch_count = 1;
  const int filter_size = 3;
  const int filter_count = 16;
  const int stride_width = 1;
  const int stride_height = 1;
  const Padding padding = Padding_SAME;

  std::vector<int> input_dims = {image_batch_count, image_height, image_width, depth};
  std::vector<int> filter_dims =  {filter_count, filter_size, filter_size, depth};
  std::vector<int> bias_dims =  {filter_count, 1,1,1};

  ConvolutionOpModel m(
      {TensorType_FLOAT32,
       {image_batch_count, image_height, image_width, depth}},
      {TensorType_FLOAT32, {filter_count, filter_size, filter_size, depth}},
      {TensorType_FLOAT32, {}}, stride_width, stride_height, padding);

  m.SetInput(input_dims);
  m.SetFilter(filter_dims);
  m.SetBias(bias_dims);

  #ifdef PROF_RISCV
  tflite::riscv::stats::csr counters;
  tflite::riscv::stats::StartStats(&counters);  // enable csr counters
  #endif

  for(int i = 0; i < num_runs; i++){
    m.Invoke();
  }

  #ifdef PROF_RISCV
  tflite::riscv::stats::StopStats(&counters);    // disable csr counters
  tflite::riscv::stats::PrintStats(&counters);
  #endif
}

void ConvBenchmarkFloat32InputDepth(int matrix_size, int num_runs) {
  const int depth = matrix_size;
  const int image_width = 32;
  const int image_height = 32;
  const int image_batch_count = 1;
  const int filter_size = 3;
  const int filter_count = 16;
  const int stride_width = 1;
  const int stride_height = 1;
  const Padding padding = Padding_SAME;

  std::vector<int> input_dims = {image_batch_count, image_height, image_width, depth};
  std::vector<int> filter_dims =  {filter_count, filter_size, filter_size, depth};
  std::vector<int> bias_dims =  {filter_count, 1,1,1};

  ConvolutionOpModel m(
      {TensorType_FLOAT32,
       {image_batch_count, image_height, image_width, depth}},
      {TensorType_FLOAT32, {filter_count, filter_size, filter_size, depth}},
      {TensorType_FLOAT32, {}}, stride_width, stride_height, padding);

  m.SetInput(input_dims);
  m.SetFilter(filter_dims);
  m.SetBias(bias_dims);

  #ifdef PROF_RISCV
  tflite::riscv::stats::csr counters;
  tflite::riscv::stats::StartStats(&counters);  // enable csr counters
  #endif

  for(int i = 0; i < num_runs; i++){
    m.Invoke();
  }

  #ifdef PROF_RISCV
  tflite::riscv::stats::StopStats(&counters);    // disable csr counters
  tflite::riscv::stats::PrintStats(&counters);
  #endif
}

}  // namespace benchmark_conv
}  // namespace tflite



int main(int argc, char** argv) {
  if (argc != 3) {
    fprintf(stderr, "<binary> <matrix size> <num runs>\n");
    return 1;
  }
  int matix_size = atoi(argv[1]);
  int num_runs = atoi(argv[2]);
  tflite::benchmark_conv::ConvBenchmarkFloat32InputWidthHeight(matix_size, num_runs);
  tflite::benchmark_conv::ConvBenchmarkFloat32InputDepth(matix_size, num_runs);
}
