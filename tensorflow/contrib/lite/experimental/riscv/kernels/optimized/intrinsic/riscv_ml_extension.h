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
#ifndef TENSORFLOW_CONTRIB_LITE_EXPERIMENTAL_RISCV_KERNELS_OPTIMIZED_INTRINSIC_RISCV_ML_EXTENSION_H_
#define TENSORFLOW_CONTRIB_LITE_EXPERIMENTAL_RISCV_KERNELS_OPTIMIZED_INTRINSIC_RISCV_ML_EXTENSION_H_

#define MAX_BLOCK_SIZE 4

enum vconfig {
  kElementWidthMax8 = 0x120,
  kElementWidthMax16 = 0x220,
  kElementWidthMax32 = 0x320,
  kElementWidthMax64 = 0x420
};  // element width

// // These are 256 bit SIMD width
// enum maxvlen {
//   kMaxVectorLength8 = 32,
//   kMaxVectorLength16 = 16,
//   kMaxVectorLength32 = 8,
//   kMaxVectorLength64 = 4
// };

// // These are 256 bit SIMD width
// enum shift_offsets {
//   kMaxVectorLength8ShiftOffset = 5,
//   kMaxVectorLength16ShiftOffset = 4,
//   kMaxVectorLength32ShiftOffset = 3,
//   kMaxVectorLength64ShiftOffset = 2
// };

// These are 128 bit SIMD width
enum maxvlen {
 kMaxVectorLength8 = 16,
 kMaxVectorLength16 = 8,
 kMaxVectorLength32 = 4,
 kMaxVectorLength64 = 2
};

// These are 128 bit SIMD width
enum shift_offsets {
 kMaxVectorLength8ShiftOffset = 4,
 kMaxVectorLength16ShiftOffset = 3,
 kMaxVectorLength32ShiftOffset = 2,
 kMaxVectorLength64ShiftOffset = 1
};


inline void SetVcfg(unsigned int config) {
  asm("csrw vcfg, %0\t\n" : : "r"(config));
}

inline void SetVl(unsigned int len) { asm("csrw vl, %0\t\n" : : "r"(len)); }

inline void SetConfig(unsigned int maxew, unsigned int maxvl) {
  asm("csrw vcfg, %0 \t\n"
      "csrw vl, %1 \t\n"
      :
      : "r"(maxew), "r"(maxvl));
}

inline void __VectorLoad(const float* load_address1,
                         const float* load_address2) {
  asm volatile(
      "vlsd va1, 0(%0), v \t\n"
      "vlsd va2, 0(%1), v \t\n"
      :
      : "r"(load_address1), "r"(load_address2));
}

inline void __VectorLoadInput1(const float* load_address) {
  asm volatile("vlsd va1, 0(%0), v \t\n" : : "r"(load_address));
}

inline void __VectorLoadInput2(const float* load_address) {
  asm volatile("vlsd va2, 0(%0), v \t\n" : : "r"(load_address));
}

inline void __VectorLoadInput1(const float* load_address, int stride) {
  asm volatile("vlsd va1, 0(%0), %1, v \t\n"
               :
               : "r"(load_address), "r"(stride));
}

inline void __VectorLoadInput2(const float* load_address, int stride) {
  asm volatile("vlsd va2, 0(%0), %1, v \t\n"
               :
               : "r"(load_address), "r"(stride));
}

inline void __VectorLoadActivationInput(float* load_address) {
  asm volatile("vlsd vt11, 0(%0), v\t\n"
               :
               : "r"(load_address));
}


inline void __VectorLoadActivationInput(float* load_address, int stride) {
  asm volatile("vlsd vt11, 0(%0), %1, v \t\n"
               :
               : "r"(load_address), "r"(stride));
}


inline void __VectorLoadBias(const float* load_address) {
  asm volatile("vlsd vt4, 0(%0), v \t\n" : : "r"(load_address));
}

inline void __VectorLoadBias(const float* load_address, int stride) {
  asm volatile("vlsd vt4, 0(%0), %1, v \t\n"
               :
               : "r"(load_address), "r"(stride));
}

inline void __VectorLoadPartialOutput(float* load_address) {
  asm volatile("vlsd vt4, 0(%0), v \t\n" : : "r"(load_address));
}

inline void __VectorLoad(const float* load_address1, const float* load_address2,
                         int stride) {
  asm volatile(
      "vlsd va1, 0(%0), v \t\n"
      "vlsd va2, 0(%1), %2, v \t\n"
      :
      : "r"(load_address1), "r"(load_address2), "r"(stride));
}

inline void __VectorLoadPartialOutput(float* load_address, int stride) {
  asm volatile("vlsd vt4, 0(%0), %1, v \t\n"
               :
               : "r"(load_address), "r"(stride));
}

inline void __VectorBroadcastAccum(float accum) {
  asm volatile("vbcastx vt4, %0 \t\n" : : "r"(accum));
}

inline void __VectorBroadcastInput(int input_index) {
  asm volatile("vsplat va1, vt2, %0 \t\n" : : "r"(input_index));
}

inline void __VectorBroadcastMinMax(float value_minimum, float value_maximum) {
  asm volatile(
      "vbcastx vt2, %0 \t\n"
      "vbcastx vt3, %1 \t\n"
      :
      : "r"(value_minimum), "r"(value_maximum));
}

inline void __VectorReduceAccumFloat() {
  asm volatile("vfredsum vt11, vt4 \t\n");
}

inline void __VectorAddFloat() { asm volatile("vfadd vt11, va1, va2, v \t\n"); }

inline void __VectorMulFloat() { asm volatile("vfmul vt11, va1, va2, v \t\n"); }

inline void __VectorMulAccFloat() {
  asm volatile("vfmadd vt4, va1, va2, vt4, v \t\n");
}

inline void __VectorAccFloat() { asm volatile("vfadd vt4, va1, vt4, v \t\n"); }

inline void __VectorMinMaxFloat() {
  asm volatile(
      "vfmax vt11, vt11, vt2 \t\n"
      "vfmin vt11, vt11, vt3 \t\n");
}

inline void __VectorMergeFloat() {
  asm volatile("vmerge vt11, vt11, vt10, t \t\n");
}
inline void __VectorSetMask(unsigned int idx, unsigned int val) {
  asm volatile("vinsx vt1, %1, %0, v \t\n" : : "r"(idx), "r"(val));
}

inline void __VectorResetMask(unsigned int idx) {
  asm volatile("vinsx vt1, zero, %0, v \t\n" : : "r"(idx));
}

inline void __VectorResetMaskAll() { asm volatile("vbcastx vt1, zero \t\n"); }

inline void __VectorStore(float* store_address) {
  asm volatile("vssd vt11, 0(%0), v \t\n" : : "r"(store_address));
}

inline void __VectorStorePartialOutput(float* store_address, int stride) {
  asm volatile("vssd vt4, 0(%0), %1, v \t\n"
               :
               : "r"(store_address), "r"(stride));
}

inline void __VectorStorePartialOutput(float* store_address) {
  asm volatile("vssd vt4, 0(%0), v \t\n" : : "r"(store_address));
}

inline void __VectorStoreAccum(float* store_address) {
  asm volatile("vssd vt11, 0(%0), s \t\n" : : "r"(store_address));
}

inline void __VectorSplatMulAccFloat(int idx) {
  asm volatile("vsplat vs3, va1, %0, v \t\n"
               "vfmadd vt4, vs3, va2, vt4, v \t\n"
               :
               : "r"(idx)
               );
}


inline void __VectorLoadInput1Block1(const float* load_address) {
  asm volatile("vlsd va1, 0(%0), s \t\n"
               :
               : "r"(load_address));
}

inline void __VectorLoadInput2Block1(const float* load_address, int stride) {
  asm volatile("vlsd vs1, 0(%0), %1, v \t\n"
               :
               : "r"(load_address), "r"(stride));
}

inline void __VectorClearOutputAccumBlock1x1() {
  asm volatile("vbcastx vt1, zero \t\n");
}

inline void __VectorMulAddFloatBlock1x1() {
  asm volatile("vfmadd vt1, va1, vs1, vt1, v \t\n");
}

inline void __VectorStoreOutputBlock1x1(float* store_address) {
  asm volatile("vssd vt1, 0(%0), v \t\n"
               :
               : "r"(store_address));
}

inline void __VectorLoadInput1Block4(const float* load_address1,
                                     const float* load_address2,
                                     const float* load_address3,
                                     const float* load_address4) {
  asm volatile("vlsd va1, 0(%0), s \t\n"
               "vlsd va2, 0(%1), s \t\n"
               "vlsd va3, 0(%2), s \t\n"
               "vlsd va4, 0(%3), s \t\n"
               :
               : "r"(load_address1), "r"(load_address2), "r"(load_address3), "r"(load_address4));
}

inline void __VectorLoadInput2Block2(const float* load_address1,
                                     const float* load_address2,
                                     int stride) {
  asm volatile("vlsd vs1, 0(%0), %2, v \t\n"
               "vlsd vs2, 0(%1), %2, v \t\n"
               :
               : "r"(load_address1), "r"(load_address2), "r" (stride));
}

inline void __VectorLoadInput2Block3(const float* load_address1,
                                     const float* load_address2,
                                     const float* load_address3,
                                     int stride) {
  asm volatile("vlsd vs1, 0(%0), %3, v \t\n"
               "vlsd vs2, 0(%1), %3, v \t\n"
               "vlsd vs3, 0(%2), %3, v \t\n"
               :
               : "r"(load_address1), "r"(load_address2), "r"(load_address3), "r"(stride));
}

inline void __VectorLoadInput2Block4(const float* load_address1,
                                     const float* load_address2,
                                     const float* load_address3,
                                     const float* load_address4,
                                     int stride) {
  asm volatile("vlsd vs1, 0(%0), %4, v \t\n"
               "vlsd vs2, 0(%1), %4, v \t\n"
               "vlsd vs3, 0(%2), %4, v \t\n"
               "vlsd vs4, 0(%3), %4, v \t\n"
               :
               : "r"(load_address1), "r"(load_address2), "r"(load_address3), "r"(load_address4), "r"(stride));
}

inline void __VectorClearOutputAccumBlock4x1() {
  asm volatile("vbcastx vt1, zero \t\n"
               "vbcastx vt2, zero \t\n"
               "vbcastx vt3, zero \t\n"
               "vbcastx vt4, zero \t\n");
}

inline void __VectorMulAddFloatBlock4x1() {
  asm volatile("vfmadd vt1, va1, vs1, vt1, v \t\n"
               "vfmadd vt2, va2, vs1, vt2, v \t\n"
               "vfmadd vt3, va3, vs1, vt3, v \t\n"
               "vfmadd vt4, va4, vs1, vt4, v \t\n");
}

inline void __VectorStoreOutputBlock4x1(float* store_address1,
                                      float* store_address2,
                                      float* store_address3,
                                      float* store_address4) {
  asm volatile("vssd vt1, 0(%0), v \t\n"
               "vssd vt2, 0(%1), v \t\n"
               "vssd vt3, 0(%2), v \t\n"
               "vssd vt4, 0(%3), v \t\n"
               :
               : "r"(store_address1), "r"(store_address2), "r"(store_address3), "r"(store_address4));
}

inline void __VectorClearOutputAccumBlock1x2() {
  asm volatile("vbcastx vt1, zero \t\n"
               "vbcastx vt5, zero \t\n");
}

inline void __VectorMulAddFloatBlock1x2() {
  asm volatile("vfmadd vt1, va1, vs1, vt1, v \t\n"
               "vfmadd vt5, va1, vs2, vt2, v \t\n");
}

inline void __VectorStoreOutputBlock1x2(float* store_address1,
                                      float* store_address2) {
  asm volatile("vssd vt1, 0(%0), v \t\n"
               "vssd vt5, 0(%1), v \t\n"
               :
               : "r"(store_address1), "r"(store_address2));
}

inline void __VectorClearOutputAccumBlock4x2() {
  asm volatile("vbcastx vt1, zero \t\n"
               "vbcastx vt2, zero \t\n"
               "vbcastx vt3, zero \t\n"
               "vbcastx vt4, zero \t\n"
               "vbcastx vt5, zero \t\n"
               "vbcastx vt6, zero \t\n"
               "vbcastx vt7, zero \t\n"
               "vbcastx vt8, zero \t\n");
}

inline void __VectorMulAddFloatBlock4x2() {
  asm volatile("vfmadd vt1, va1, vs1, vt1, v \t\n"
               "vfmadd vt2, va2, vs1, vt2, v \t\n"
               "vfmadd vt3, va3, vs1, vt3, v \t\n"
               "vfmadd vt4, va4, vs1, vt4, v \t\n"
               "vfmadd vt5, va1, vs2, vt5, v \t\n"
               "vfmadd vt6, va2, vs2, vt6, v \t\n"
               "vfmadd vt7, va3, vs2, vt7, v \t\n"
               "vfmadd vt8, va4, vs2, vt8, v \t\n");
}

inline void __VectorStoreOutputBlock4x2(float* store_address1,
                                      float* store_address2,
                                      float* store_address3,
                                      float* store_address4,
                                      float* store_address5,
                                      float* store_address6,
                                      float* store_address7,
                                      float* store_address8) {
  asm volatile("vssd vt1, 0(%0), v \t\n"
               "vssd vt2, 0(%1), v \t\n"
               "vssd vt3, 0(%2), v \t\n"
               "vssd vt4, 0(%3), v \t\n"
               "vssd vt5, 0(%4), v \t\n"
               "vssd vt6, 0(%5), v \t\n"
               "vssd vt7, 0(%6), v \t\n"
               "vssd vt8, 0(%7), v \t\n"
               :
               : "r"(store_address1), "r"(store_address2), "r"(store_address3), "r"(store_address4),
                 "r"(store_address5), "r"(store_address6), "r"(store_address7), "r"(store_address8));
}


inline void __VectorClearOutputAccumBlock1x3() {
  asm volatile("vbcastx vt1, zero \t\n"
               "vbcastx vt5, zero \t\n"
               "vbcastx vt9, zero \t\n");
}

inline void __VectorMulAddFloatBlock1x3() {
  asm volatile("vfmadd vt1, va1, vs1, vt1, v \t\n"
               "vfmadd vt5, va1, vs2, vt5, v \t\n"
               "vfmadd vt9, va1, vs3, vt9, v \t\n");
}


inline void __VectorStoreOutputBlock1x3(float* store_address1,
                                       float* store_address2,
                                       float* store_address3) {
  asm volatile("vssd vt1, 0(%0), v \t\n"
               "vssd vt5, 0(%1), v \t\n"
               "vssd vt9, 0(%2), v \t\n"
               :
               : "r"(store_address1), "r"(store_address2), "r"(store_address3));
}

inline void __VectorClearOutputAccumBlock4x3() {
  asm volatile("vbcastx vt1, zero \t\n"
               "vbcastx vt2, zero \t\n"
               "vbcastx vt3, zero \t\n"
               "vbcastx vt4, zero \t\n"
               "vbcastx vt5, zero \t\n"
               "vbcastx vt6, zero \t\n"
               "vbcastx vt7, zero \t\n"
               "vbcastx vt8, zero \t\n"
               "vbcastx vt9, zero \t\n"
               "vbcastx vt10, zero \t\n"
               "vbcastx vt11, zero \t\n"
               "vbcastx vs11, zero \t\n");
}

inline void __VectorMulAddFloatBlock4x3() {
  asm volatile("vfmadd vt1, va1, vs1, vt1, v \t\n"
               "vfmadd vt2, va2, vs1, vt2, v \t\n"
               "vfmadd vt3, va3, vs1, vt3, v \t\n"
               "vfmadd vt4, va4, vs1, vt4, v \t\n"
               "vfmadd vt5, va1, vs2, vt5, v \t\n"
               "vfmadd vt6, va2, vs2, vt6, v \t\n"
               "vfmadd vt7, va3, vs2, vt7, v \t\n"
               "vfmadd vt8, va4, vs2, vt8, v \t\n"
               "vfmadd vt9, va1, vs3, vt9, v \t\n"
               "vfmadd vt10, va2, vs3, vt10, v \t\n"
               "vfmadd vt11, va3, vs3, vt11, v \t\n"
               "vfmadd vs11, va4, vs3, vs11, v \t\n");
}


inline void __VectorStoreOutputBlock4x3(float* store_address1,
                                       float* store_address2,
                                       float* store_address3,
                                       float* store_address4,
                                       float* store_address5,
                                       float* store_address6,
                                       float* store_address7,
                                       float* store_address8,
                                       float* store_address9,
                                       float* store_address10,
                                       float* store_address11,
                                       float* store_address12) {
  asm volatile("vssd vt1, 0(%0), v \t\n"
               "vssd vt2, 0(%1), v \t\n"
               "vssd vt3, 0(%2), v \t\n"
               "vssd vt4, 0(%3), v \t\n"
               "vssd vt5, 0(%4), v \t\n"
               "vssd vt6, 0(%5), v \t\n"
               "vssd vt7, 0(%6), v \t\n"
               "vssd vt8, 0(%7), v \t\n"
               "vssd vt9, 0(%8), v \t\n"
               "vssd vt10, 0(%9), v \t\n"
               "vssd vt11, 0(%10), v \t\n"
               "vssd vs11, 0(%11), v \t\n"
               :
               : "r"(store_address1), "r"(store_address2), "r"(store_address3), "r"(store_address4),
                 "r"(store_address5), "r"(store_address6), "r"(store_address7), "r"(store_address8),
                 "r"(store_address9), "r"(store_address10), "r"(store_address11), "r"(store_address12));
}

inline void __VectorClearOutputAccumBlock1x4() {
  asm volatile("vbcastx vt1, zero \t\n"
               "vbcastx vt5, zero \t\n"
               "vbcastx vt9, zero \t\n"
               "vbcastx vt10, zero \t\n");
}

inline void __VectorMulAddFloatBlock1x4() {
  asm volatile("vfmadd vt1, va1, vs1, vt1, v \t\n"
               "vfmadd vt5, va1, vs2, vt5, v \t\n"
               "vfmadd vt9, va1, vs3, vt9, v \t\n"
               "vfmadd vs10, va1, vs4, vs10, v \t\n");
}
inline void __VectorStoreOutputBlock1x4(float* store_address1,
                                       float* store_address2,
                                       float* store_address3,
                                       float* store_address4) {
  asm volatile("vssd vt1, 0(%0), v \t\n"
               "vssd vt5, 0(%1), v \t\n"
               "vssd vt9, 0(%2), v \t\n"
               "vssd vs10, 0(%3), v \t\n"
               :
               : "r"(store_address1), "r"(store_address2), "r"(store_address3), "r"(store_address4));
}

inline void __VectorClearOutputAccumBlock4x4() {
  asm volatile("vbcastx vt1, zero \t\n"
               "vbcastx vt2, zero \t\n"
               "vbcastx vt3, zero \t\n"
               "vbcastx vt4, zero \t\n"
               "vbcastx vt5, zero \t\n"
               "vbcastx vt6, zero \t\n"
               "vbcastx vt7, zero \t\n"
               "vbcastx vt8, zero \t\n"
               "vbcastx vt9, zero \t\n"
               "vbcastx vt10, zero \t\n"
               "vbcastx vt11, zero \t\n"
               "vbcastx vs11, zero \t\n"
               "vbcastx vs10, zero \t\n"
               "vbcastx vs9, zero \t\n"
               "vbcastx vs8, zero \t\n"
               "vbcastx vs7, zero \t\n");
}

inline void __VectorMulAddFloatBlock4x4() {
  asm volatile("vfmadd vt1, va1, vs1, vt1, v \t\n"
               "vfmadd vt2, va2, vs1, vt2, v \t\n"
               "vfmadd vt3, va3, vs1, vt3, v \t\n"
               "vfmadd vt4, va4, vs1, vt4, v \t\n"
               "vfmadd vt5, va1, vs2, vt5, v \t\n"
               "vfmadd vt6, va2, vs2, vt6, v \t\n"
               "vfmadd vt7, va3, vs2, vt7, v \t\n"
               "vfmadd vt8, va4, vs2, vt8, v \t\n"
               "vfmadd vt9, va1, vs3, vt9, v \t\n"
               "vfmadd vt10, va2, vs3, vt10, v \t\n"
               "vfmadd vt11, va3, vs3, vt11, v \t\n"
               "vfmadd vs11, va4, vs3, vs11, v \t\n"
               "vfmadd vs10, va1, vs4, vs10, v \t\n"
               "vfmadd vs9, va2, vs4, vs9, v \t\n"
               "vfmadd vs8, va3, vs4, vs8, v \t\n"
               "vfmadd vs7, va4, vs4, vs7, v \t\n");
}
inline void __VectorStoreOutputBlock4x4(float* store_address1,
                                       float* store_address2,
                                       float* store_address3,
                                       float* store_address4,
                                       float* store_address5,
                                       float* store_address6,
                                       float* store_address7,
                                       float* store_address8,
                                       float* store_address9,
                                       float* store_address10,
                                       float* store_address11,
                                       float* store_address12,
                                       float* store_address13,
                                       float* store_address14,
                                       float* store_address15,
                                       float* store_address16) {
  asm volatile("vssd vt1, 0(%0), v \t\n"
               "vssd vt2, 0(%1), v \t\n"
               "vssd vt3, 0(%2), v \t\n"
               "vssd vt4, 0(%3), v \t\n"
               "vssd vt5, 0(%4), v \t\n"
               "vssd vt6, 0(%5), v \t\n"
               "vssd vt7, 0(%6), v \t\n"
               "vssd vt8, 0(%7), v \t\n"
               "vssd vt9, 0(%8), v \t\n"
               "vssd vt10, 0(%9), v \t\n"
               "vssd vt11, 0(%10), v \t\n"
               "vssd vs11, 0(%11), v \t\n"
               "vssd vs10, 0(%12), v \t\n"
               "vssd vs9, 0(%13), v \t\n"
               "vssd vs8, 0(%14), v \t\n"
               "vssd vs7, 0(%15), v \t\n"
               :
               : "r"(store_address1), "r"(store_address2), "r"(store_address3), "r"(store_address4),
                 "r"(store_address5), "r"(store_address6), "r"(store_address7), "r"(store_address8),
                 "r"(store_address9), "r"(store_address10), "r"(store_address11), "r"(store_address12),
                 "r"(store_address13), "r"(store_address14), "r"(store_address15), "r"(store_address16));
}

// template <int block_size>
// struct MatrixMatrixBlock {};

// template <>
// struct

void VectorVectorAdd(const float* input1, const float* input2, float* output,
                     int len);

void VectorVectorAddMinMax(const float* input1, const float* input2,
                           float output_min, float output_max, float* output,
                           int len);

void VectorVectorMultiply(const float* input1, const float* input2,
                          float* output, int len);

void VectorVectorMultiplyAccumulate(const float* input1, const float* input2,
                                    float* output, int len);

void VectorMatrixMultiplyAccumulate(const float* vector, const float* matrix,
                                    float* output, int matrix_rows,
                                    int matrix_cols);

void VectorVectorMultiplyAccumulateDepthwise(const float* input1,
                                             const float* input2, float* output,
                                             int depth, int depth_multiplier,
                                             const float* bias, bool use_bias);

void VectorAveragePooling(const float* input, float* output, int depth,
                          bool use_zero);

void VectorActivationFunctionWithMinMax(float* data, float activation_min,
                                        float activation_max, int length);


void MatrixVectorMultiplyAccumulate(const float* matrix, int matrix_rows,
                                    int matrix_cols, const float* vector,
                                    float* output);

void MatrixBatchVectorMultiplyAccumulate(const float* matrix, int matrix_rows,
                                         int matrix_cols, const float* vector,
                                         int batch_size, float* output,
                                         int output_stride);

void VectorBatchVectorCwiseProductAccumulate(const float* vector,
                                             int v_len,
                                             const float* batch_vector,
                                             int batch_size,
                                             float* output);
void AddBiasActivationFunctionWithMinMax(float* vector, const float* bias,
                                         float output_min, float output_max,
                                         int flatted_len, int bias_len);


void Kernel1x1MultiplyAccumulate(const float* filter, int input_depth,
                                 int output_depth,
                                 const float* input,
                                 float* output);

void MatrixMatrixMultiplyAccumulate4x1(const float* matrix_a,
                                       int matrix_a_rows,
                                       int matrix_a_cols,
                                       const float* matrix_b,
                                       int matrix_b_rows,
                                       int matrix_b_cols,
                                       float* matrix_c);


void MatrixMatrixMultiplyAccumulate4x2(const float* matrix_a,
                                       int matrix_a_rows,
                                       int matrix_a_cols,
                                       const float* matrix_b,
                                       int matrix_b_rows,
                                       int matrix_b_cols,
                                       float* matrix_c);


void MatrixMatrixMultiplyAccumulate4x3(const float* matrix_a,
                                       int matrix_a_rows,
                                       int matrix_a_cols,
                                       const float* matrix_b,
                                       int matrix_b_rows,
                                       int matrix_b_cols,
                                       float* matrix_c);

void MatrixMatrixMultiplyAccumulate(const float* matrix_a,
                                    int matrix_a_rows,
                                    int matrix_a_cols,
                                    const float* matrix_b,
                                    int matrix_b_rows,
                                    int matrix_b_cols,
                                    float* matrix_c);
#endif  // TENSORFLOW_CONTRIB_LITE_EXPERIMENTAL_RISCV_KERNELS_OPTIMIZED_INTRINSIC_RISCV_ML_EXTENSION_H_
