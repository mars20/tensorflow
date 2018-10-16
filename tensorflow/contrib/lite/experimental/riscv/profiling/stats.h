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
#ifndef TENSORFLOW_CONTRIB_LITE_EXPERIMENTAL_RISCV_PROFILING_STATS_H_
#define TENSORFLOW_CONTRIB_LITE_EXPERIMENTAL_RISCV_PROFILING_STATS_H_

# include <stdio.h>

#define read_csr(reg) ({ unsigned long __tmp;    \
  asm volatile ("csrr %0, " #reg : "=r"(__tmp)); \
   __tmp; })

namespace tflite {

namespace riscv {

namespace stats {

struct csr{
  unsigned long cycles = 0;
  unsigned long instret = 0;
};

inline void StartStats(struct csr *counters) {
  counters->cycles = read_csr(cycle);
  counters->instret = read_csr(instret);
}

inline void StopStats(struct csr *counters) {
  counters->instret = read_csr(instret) - counters->instret;
  counters->cycles = read_csr(cycle) - counters->cycles;
}

inline void PrintStats(struct csr *counters) {
  printf("# Cycles in ROI = %ld\n", counters->cycles);
  printf("# Instruction in ROI = %ld\n", counters->instret);
}

} // stats
} // riscv
} // tflite


#endif  // TENSORFLOW_CONTRIB_LITE_EXPERIMENTAL_RISCV_PROFILING_STATS_H_
