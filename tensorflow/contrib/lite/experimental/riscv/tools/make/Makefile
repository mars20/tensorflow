# Find where we're running from, so we can store generated files here.
ifeq ($(origin MAKEFILE_DIR), undefined)
	MAKEFILE_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
endif

# Try to figure out the host system
HOST_OS :=
ifeq ($(OS),Windows_NT)
	HOST_OS = windows
else
	UNAME_S := $(shell uname -s)
	ifeq ($(UNAME_S),Linux)
		HOST_OS := linux
	endif
	ifeq ($(UNAME_S),Darwin)
		HOST_OS := osx
	endif
endif

HOST_ARCH := $(shell if [[ $(shell uname -m) =~ i[345678]86 ]]; then echo x86_32; else echo $(shell uname -m); fi)

# Override these on the make command line to target a specific architecture. For example:
# make -f tensorflow/contrib/lite/experimental/riscv/tools/make/Makefile TARGET=riscv
TARGET := $(HOST_OS)
TARGET_ARCH := $(HOST_ARCH)

# These are the default libraries needed, but they can be added to or
# overridden by the platform-specific settings in target makefiles.
LIBS := -lstdc++

# There are no rules for compiling objects for the host system (since we don't
# generate things like the protobuf compiler that require that), so all of
# these settings are for the target compiler.
CXXFLAGS := -O3 -DNDEBUG
CCFLAGS := ${CXXFLAGS}
CXXFLAGS += --std=gnu++11
CFLAGS :=
LDOPTS := -L/usr/local/lib
ARFLAGS := -r
TARGET_TOOLCHAIN_PREFIX :=
CC_PREFIX :=

# These target-specific makefiles should modify or replace options like
# CXXFLAGS or LIBS to work for a specific targetted architecture. All logic
# based on platforms or architectures should happen within these files, to
# keep this main makefile focused on the sources and dependencies.
include $(wildcard $(MAKEFILE_DIR)/targets/*_makefile.inc)

# Where compiled objects are stored.
GENDIR := $(MAKEFILE_DIR)/gen/$(TARGET)_$(TARGET_ARCH)/
OBJDIR := $(GENDIR)obj/
BINDIR := $(GENDIR)bin/
LIBDIR := $(GENDIR)lib/


# TODO(scrub the dependencies)
INCLUDES := \
-I. \
-I$(MAKEFILE_DIR)/../../../../../ \
-I$(MAKEFILE_DIR)/../../../../../../ \
-I$(MAKEFILE_DIR)/downloads/ \
-I$(MAKEFILE_DIR)/downloads/eigen \
-I$(MAKEFILE_DIR)/downloads/gemmlowp \
-I$(MAKEFILE_DIR)/downloads/neon_2_sse \
-I$(MAKEFILE_DIR)/downloads/flatbuffers/include \
-I$(OBJDIR)
# This is at the end so any globally-installed frameworks like protobuf don't
# override local versions in the source tree.
INCLUDES += -I/usr/local/include

CXX := $(CC_PREFIX)${TARGET_TOOLCHAIN_PREFIX}g++
CC := $(CC_PREFIX)${TARGET_TOOLCHAIN_PREFIX}gcc
AR := $(CC_PREFIX)${TARGET_TOOLCHAIN_PREFIX}ar

# This library is the main target for this makefile. It will contain a minimal
# runtime that can be linked in to other programs.
LIB_NAME := libtensorflow-lite.a
LIB_PATH := $(LIBDIR)$(LIB_NAME)

# A small example program that shows how to link against the library.
MINIMAL_PATH := $(BINDIR)minimal

# Benchmark static library and binary
#BENCHMARK_LIB_NAME := benchmark-lib.a
#BENCHMARK_BINARY_NAME := benchmark_model
#BENCHMARK_LIB := $(LIBDIR)$(BENCHMARK_LIB_NAME)
#BENCHMARK_BINARY := $(BINDIR)$(BENCHMARK_BINARY_NAME)

PROFILER_SRCS := \
	$(GOOGLE3)/third_party/tensorflow/contrib/lite/profiling/time.cc
PROFILE_SUMMARIZER_SRCS := \
	$(GOOGLE3)/third_party/tensorflow/contrib/lite/profiling/profile_summarizer.cc \
	$(GOOGLE3)/third_party/tensorflow/core/util/stats_calculator.cc

MINIMAL_SRCS := \
experimental/users/suyoggupta/riscv/minimal/minimal.cc \
$(PROFILER_SRCS)
MINIMAL_OBJS := $(addprefix $(OBJDIR), \
$(patsubst %.cc,%.o,$(patsubst %.c,%.o,$(MINIMAL_SRCS))))

# What sources we want to compile, must be kept in sync with the main Bazel
# build files.



CORE_CC_ALL_SRCS := \
$(wildcard $(GOOGLE3)/third_party/tensorflow/contrib/lite/*.cc) \
$(wildcard $(GOOGLE3)/third_party/tensorflow/contrib/lite/*.c) \
$(wildcard $(GOOGLE3)/experimental/users/suyoggupta/riscv/kernels/*.cc) \
$(GOOGLE3)/third_party/tensorflow/contrib/lite/kernels/kernel_util.cc \
$(GOOGLE3)/third_party/tensorflow/contrib/lite/kernels/internal/tensor_utils.cc \
$(GOOGLE3)/third_party/tensorflow/contrib/lite/kernels/internal/kernel_utils.cc \
$(GOOGLE3)/third_party/tensorflow/contrib/lite/kernels/internal/reference/portable_tensor_utils.cc \
# Remove any duplicates.
CORE_CC_ALL_SRCS := $(sort $(CORE_CC_ALL_SRCS))
CORE_CC_EXCLUDE_SRCS := \
$(wildcard $(GOOGLE3)/third_party/tensorflow/contrib/lite/*test.cc) \
$(wildcard $(GOOGLE3)/third_party/tensorflow/contrib/lite/*/*test.cc) \
$(wildcard $(GOOGLE3)/third_party/tensorflow/contrib/lite/*/*/*test.cc) \
$(wildcard $(GOOGLE3)/third_party/tensorflow/contrib/lite/*/*/*/*test.cc) \
$(wildcard $(GOOGLE3)/third_party/tensorflow/contrib/lite/kernels/test_util.cc) \
$(MINIMAL_SRCS)
CORE_CC_EXCLUDE_SRCS += \
$(GOOGLE3)/third_party/tensorflow/contrib/lite/mmap_allocation.cc \
$(GOOGLE3)/third_party/tensorflow/contrib/lite/nnapi_delegate.cc

# Filter out all the excluded files.
TF_LITE_CC_SRCS := $(filter-out $(CORE_CC_EXCLUDE_SRCS), $(CORE_CC_ALL_SRCS))
# File names of the intermediate files target compilation generates.
TF_LITE_CC_OBJS := $(addprefix $(OBJDIR), \
$(patsubst %.cc,%.o,$(patsubst %.c,%.o,$(TF_LITE_CC_SRCS))))
LIB_OBJS := $(TF_LITE_CC_OBJS)

# Benchmark sources
#BENCHMARK_SRCS_DIR := $(GOOGLE3)/third_party/tensorflow/contrib/lite/tools/benchmark
#BENCHMARK_ALL_SRCS := $(TFLITE_CC_SRCS) \
#	$(wildcard $(BENCHMARK_SRCS_DIR)/*.cc) \
#	$(PROFILE_SUMMARIZER_SRCS)

#BENCHMARK_SRCS := $(filter-out \
#	$(wildcard $(BENCHMARK_SRCS_DIR)/*_test.cc), \
#    $(BENCHMARK_ALL_SRCS))

#BENCHMARK_OBJS := $(addprefix $(OBJDIR), \
#$(patsubst %.cc,%.o,$(patsubst %.c,%.o,$(BENCHMARK_SRCS))))

# For normal manually-created TensorFlow C++ source files.
$(OBJDIR)%.o: %.cc
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@
# For normal manually-created TensorFlow C source files.
$(OBJDIR)%.o: %.c
	@mkdir -p $(dir $@)
	$(CC) $(CCFLAGS) $(INCLUDES) -c $< -o $@

# The target that's compiled if there's no command-line arguments.
ifeq ($(TARGET),RV64G)
  all: HACK_TFLITE_FOR_RISCV $(LIB_PATH) $(MINIMAL_PATH)
else
  all: $(LIB_PATH) $(MINIMAL_PATH)
endif

# Hack for generating schema file bypassing flatbuffer parsing
GENERATE_SCHEMA:
	@cp -u $(GOOGLE3)/third_party/tensorflow/contrib/lite/schema/schema_generated.h.OPENSOURCE $(GOOGLE3)/third_party/tensorflow/contrib/lite/schema/schema_generated.h


COPY_FILES:
	@cp $(GOOGLE3)/experimental/users/suyoggupta/riscv/allocation.* $(GOOGLE3)/third_party/tensorflow/contrib/lite/
	@cp $(GOOGLE3)/experimental/users/suyoggupta/riscv/common.h $(GOOGLE3)/third_party/tensorflow/contrib/lite/kernels/internal/
	@cp $(GOOGLE3)/experimental/users/suyoggupta/riscv/round.h $(GOOGLE3)/third_party/tensorflow/contrib/lite/kernels/internal/


# Hacks to make base tflite work for riscv:
HACK_TFLITE_FOR_RISCV: $(info "Building for RV64G") COPY_FILES

# Gathers together all the objects we've compiled into a single '.a' archive.
$(LIB_PATH): HACK_TFLITE_FOR_RISCV GENERATE_SCHEMA $(LIB_OBJS)
	@mkdir -p $(dir $@)
	$(AR) $(ARFLAGS) $(LIB_PATH) $(LIB_OBJS)

$(MINIMAL_PATH): $(MINIMAL_OBJS) $(LIB_PATH)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) \
	-o $(MINIMAL_PATH) $(MINIMAL_OBJS) \
	$(LIBFLAGS) $(LIB_PATH) $(LDFLAGS) $(LIBS)


$(BENCHMARK_LIB) : $(LIB_PATH) $(BENCHMARK_OBJS)
	@mkdir -p $(dir $@)
	$(AR) $(ARFLAGS) $(BENCHMARK_LIB) $(LIB_OBJS) $(BENCHMARK_OBJS)

benchmark_lib: $(BENCHMARK_LIB)
$(info $(BENCHMARK_BINARY))
$(BENCHMARK_BINARY) : $(BENCHMARK_LIB)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) \
	-o $(BENCHMARK_BINARY) \
	$(LIBFLAGS) $(BENCHMARK_LIB) $(LDFLAGS) $(LIBS)

benchmark: $(BENCHMARK_BINARY)

# Gets rid of all generated files.
clean:
	rm -rf $(MAKEFILE_DIR)/gen

# Gets rid of target files only, leaving the host alone. Also leaves the lib
# directory untouched deliberately, so we can persist multiple architectures
# across builds for iOS and Android.
cleantarget:
	rm -rf $(OBJDIR)
	rm -rf $(BINDIR)

$(DEPDIR)/%.d: ;
.PRECIOUS: $(DEPDIR)/%.d

-include $(patsubst %,$(DEPDIR)/%.d,$(basename $(TF_CC_SRCS)))
