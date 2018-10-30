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
#include <cstdio>
#include "tensorflow/contrib/lite/experimental/riscv/kernels/register.h"
#include "tensorflow/contrib/lite/interpreter.h"
#include "tensorflow/contrib/lite/model.h"
#include "tensorflow/contrib/lite/op_resolver.h"
//#include "tensorflow/contrib/lite/optional_debug_tools.h"
#include "tensorflow/contrib/lite/profiling/time.h"
#include "tensorflow/contrib/lite/experimental/riscv/profiling/stats.h"

// Usage: minimal <tflite model> <num runs>
#ifdef ARM_GEM5
extern "C" void m5_checkpoint(int a, int b);
extern "C" void m5_reset_stats(int a, int b);
extern "C" void m5_dump_stats (int a, int b);
extern "C" void m5_exit (int a);
#endif
using namespace tflite;

#define TFLITE_MINIMAL_CHECK(x)                              \
  if (!(x)) {                                                \
    fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
    exit(1);                                                 \
  }

int main(int argc, char* argv[]) {
  int num_times;

  if (argc < 2) {
    fprintf(stderr, "minimal <tflite model> <num runs> \n");
    return 1;
  }
  if (argc == 3) {
    num_times = atoi(argv[2]);
  } else {
    num_times = 1;
  }

  const char* filename = argv[1];

  // Load model
  std::unique_ptr<tflite::FlatBufferModel> model =
      tflite::FlatBufferModel::BuildFromFile(filename);
  TFLITE_MINIMAL_CHECK(model != nullptr);

  // Build the interpreter
  tflite::ops::riscv::RiscvOpResolver resolver;
  InterpreterBuilder builder(*model.get(), resolver);
  std::unique_ptr<Interpreter> interpreter;
  TFLITE_MINIMAL_CHECK(builder(&interpreter) == kTfLiteOk);

  TFLITE_MINIMAL_CHECK(interpreter != nullptr);

  // Allocate tensor buffers
  TFLITE_MINIMAL_CHECK(interpreter->AllocateTensors() == kTfLiteOk);

  printf("=== Run started ===\n");
  int64_t start_us = profiling::time::NowMicros();
  #ifdef PROF_RISCV
  tflite::riscv::stats::csr counters;
  tflite::riscv::stats::StartStats(&counters);  // enable csr counters
  #endif


  // Fill input buffers
  // TODO(user): Insert code to fill input tensors

  // Run inference
  #ifdef ARM_GEM5
  m5_reset_stats(0,0);
  #endif

  for (int run = 0; run < num_times; run++) {
    TFLITE_MINIMAL_CHECK(interpreter->Invoke() == kTfLiteOk);
  }
  #ifdef ARM_GEM5
   m5_dump_stats(0, 0);
  #endif

  #ifdef PROF_RISCV
  tflite::riscv::stats::StopStats(&counters);    // disable csr counters
  tflite::riscv::stats::PrintStats(&counters);
  #endif
  int64_t end_us = profiling::time::NowMicros();

  printf("\nInference time = %d us\n", end_us - start_us);
  printf("=== Run complete ===\n");

  // Read output buffers
  // TODO(user): Insert getting data out code.

  return 0;
}
