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

#ifndef USE_NEON
#include "tensorflow/contrib/lite/experimental/riscv/kernels/register.h"
#else
#include "tensorflow/contrib/lite/kernels/register.h"
#endif

#include "tensorflow/contrib/lite/experimental/riscv/kernels/test_util.h"
#include "tensorflow/contrib/lite/interpreter.h"
#include "tensorflow/contrib/lite/model.h"

#ifdef PROF_RISCV
#include "tensorflow/contrib/lite/experimental/riscv/profiling/stats.h"
#endif

#ifdef ARM_GEM5
extern "C" void m5_checkpoint(int a, int b);
extern "C" void m5_reset_stats(int a, int b);
extern "C" void m5_dump_stats (int a, int b);
extern "C" void m5_exit (int a);
#endif


namespace tflite {
namespace benchmark_depthwiseconv {

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

void DepthwiseConvBenchmarkFloat32(int matrix_size, int matrix_depth,
                                   int kernel_size, int kernel_count,
                                   int num_runs) {
  const int depth = matrix_depth;
  const int image_width = matrix_size;
  const int image_height = matrix_size;
  const int image_batch_count = 1;
  const int filter_size = kernel_size;
  const int filter_count = kernel_count;
  const Padding padding = Padding_SAME;
  const int dilation_factor = 1;
  DepthwiseConvolutionOpModel m(
      {TensorType_FLOAT32,
       {image_batch_count, image_height, image_width, depth}},
      {TensorType_FLOAT32, {depth, filter_size, filter_size, filter_count}},
      {TensorType_FLOAT32, {}}, Padding_SAME, dilation_factor);

  std::vector<int> input_dims = {image_batch_count, image_height, image_width, depth};
  std::vector<int> filter_dims =  {depth, filter_size, filter_size, filter_count};
  std::vector<int> bias_dims =  {filter_count, 1,1,1};

  m.SetInput(input_dims);
  m.SetFilter(filter_dims);
  m.SetBias(bias_dims);

  #ifdef PROF_RISCV
  tflite::riscv::stats::csr counters;
  tflite::riscv::stats::StartStats(&counters);  // enable csr counters
  #endif
  #ifdef ARM_GEM5
  m5_reset_stats(0,0);
  #endif

  for(int i = 0; i < num_runs; i++){
    m.Invoke();
  }
  #ifdef PROF_RISCV
  tflite::riscv::stats::StopStats(&counters);    // disable csr counters
  tflite::riscv::stats::PrintStats(&counters);
  #endif
  #ifdef ARM_GEM5
  m5_dump_stats(0, 0);
  #endif
}

}  // namespace depthwiseconv_test
}  // namespace tflite


int main(int argc, char** argv) {
  if (argc != 6) {
    fprintf(stderr, "<binary> <matrix size> <matrix_depth> <kernel size> <kernel count> <num runs>\n");
    return 1;
  }
  int matix_size = atoi(argv[1]);
  int matix_depth = atoi(argv[2]);
  int kernel_size = atoi(argv[3]);
  int kernel_count = atoi(argv[4]);
  int num_runs = atoi(argv[5]);
  tflite::benchmark_depthwiseconv::DepthwiseConvBenchmarkFloat32(matix_size, matix_depth, kernel_size, kernel_count, num_runs);
}
