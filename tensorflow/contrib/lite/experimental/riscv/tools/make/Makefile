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
# make -f tensorflow/contrib/lite/Makefile TARGET=rpi TARGET_ARCH=armv7l
TARGET := $(HOST_OS)
TARGET_ARCH := $(HOST_ARCH)

INCLUDES := \
-I. \
-I$(MAKEFILE_DIR)/../../../../../ \
-I$(MAKEFILE_DIR)/../../../../../../ \
-I$(MAKEFILE_DIR)/downloads/ \
-I$(MAKEFILE_DIR)/downloads/eigen \
-I$(MAKEFILE_DIR)/downloads/absl \
-I$(MAKEFILE_DIR)/downloads/gemmlowp \
-I$(MAKEFILE_DIR)/downloads/neon_2_sse \
-I$(MAKEFILE_DIR)/downloads/flatbuffers/include \
-I$(OBJDIR)
# This is at the end so any globally-installed frameworks like protobuf don't
# override local versions in the source tree.
INCLUDES += -I/usr/local/include

# These are the default libraries needed, but they can be added to or
# overridden by the platform-specific settings in target makefiles.
LIBS := -lstdc++

# There are no rules for compiling objects for the host system (since we don't
# generate things like the protobuf compiler that require that), so all of
# these settings are for the target compiler.
CXXFLAGS := -O3 -DNDEBUG
CCFLAGS := ${CXXFLAGS}
CXXFLAGS += --std=c++11
CFLAGS :=
LDOPTS := -L/usr/local/lib
ARFLAGS := -r
TARGET_TOOLCHAIN_PREFIX :=
CC_PREFIX :=

# This library is the main target for this makefile. It will contain a minimal
# runtime that can be linked in to other programs.
LIB_NAME := libtensorflow-lite.a

# A small example program that loads and runs a tflite model
MINIMAL_SRCS := \
tensorflow/contrib/lite/experimental/riscv/examples/minimal.cc

#GEM5_SRCS := \
#tensorflow/contrib/lite/examples/minimal/m5op_arm_A64.S

# Test sources
TEST_SRCS := \
$(wildcard tensorflow/contrib/lite/experimental/riscv/*test.cc) \
$(wildcard tensorflow/contrib/lite/experimental/riscv/kernels/*test.cc)

# Test sources
BENCHMARK_SRCS := \
$(wildcard tensorflow/contrib/lite/experimental/riscv/benchmark/*/*.cc)

# What sources we want to compile, must be kept in sync with the main Bazel
# build files.

PROFILER_SRCS := \
	tensorflow/contrib/lite/profiling/time.cc
PROFILE_SUMMARIZER_SRCS := \
	tensorflow/contrib/lite/profiling/profile_summarizer.cc \
	tensorflow/core/util/stats_calculator.cc

CORE_CC_ALL_SRCS := \
$(wildcard tensorflow/contrib/lite/*.c) \
$(wildcard tensorflow/contrib/lite/*.cc) \
$(wildcard tensorflow/contrib/lite/c/*.c) \
$(wildcard tensorflow/contrib/lite/core/api/*.cc) \
$(wildcard tensorflow/contrib/lite/experimental/riscv/kernels/*.c) \
$(wildcard tensorflow/contrib/lite/experimental/riscv/kernels/*.cc) \
$(wildcard tensorflow/contrib/lite/experimental/riscv/kernels/reference/*.c) \
$(wildcard tensorflow/contrib/lite/experimental/riscv/kernels/reference/*.cc) \
$(wildcard tensorflow/contrib/lite/experimental/riscv/kernels/optimized/intrinsic/*.c) \
$(wildcard tensorflow/contrib/lite/experimental/riscv/kernels/optimized/intrinsic/*.cc) \
$(wildcard tensorflow/contrib/lite/experimental/riscv/kernels/test_util.cc) \
$(wildcard tensorflow/contrib/lite/kernels/internal/test_util.cc) \
$(PROFILER_SRCS)

# Remove any duplicates.
CORE_CC_ALL_SRCS := $(sort $(CORE_CC_ALL_SRCS))
CORE_CC_EXCLUDE_SRCS := \
$(wildcard tensorflow/contrib/lite/*test.cc) \
$(wildcard tensorflow/contrib/lite/*/*test.cc) \
$(wildcard tensorflow/contrib/lite/*/*/*test.cc) \
$(wildcard tensorflow/contrib/lite/*/*/*/*test.cc) \
$(wildcard tensorflow/contrib/lite/kernels/test_util.cc) \
$(MINIMAL_SRCS)

CORE_CC_EXCLUDE_SRCS += \
tensorflow/contrib/lite/mmap_allocation.cc \
tensorflow/contrib/lite/nnapi_delegate.cc

# Filter out all the excluded files.
TF_LITE_CC_SRCS := $(filter-out $(CORE_CC_EXCLUDE_SRCS), $(CORE_CC_ALL_SRCS))


# These target-specific makefiles should modify or replace options like
# CXXFLAGS or LIBS to work for a specific targetted architecture. All logic
# based on platforms or architectures should happen within these files, to
# keep this main makefile focused on the sources and dependencies.
include $(wildcard $(MAKEFILE_DIR)/targets/*_makefile.inc)

ALL_SRCS := \
	$(MINIMAL_SRCS) \
        $(TEST_SRCS) \
	$(PROFILER_SRCS) \
	$(PROFILER_SUMMARY_SRCS) \
	$(TF_LITE_CC_SRCS)

# Where compiled objects are stored.
GENDIR := $(MAKEFILE_DIR)/gen/$(TARGET)_$(TARGET_ARCH)/
OBJDIR := $(GENDIR)obj/
BINDIR := $(GENDIR)bin/
LIBDIR := $(GENDIR)lib/

LIB_PATH := $(LIBDIR)$(LIB_NAME)
MINIMAL_BINARY := $(BINDIR)minimal

TEST_BINARY := $(addprefix $(BINDIR), \
$(patsubst %_test.cc,%_test,$(TEST_SRCS)))

BENCHMARK_BINARY := $(addprefix $(BINDIR), \
$(patsubst %.cc,%,$(BENCHMARK_SRCS)))

CXX := $(CC_PREFIX)${TARGET_TOOLCHAIN_PREFIX}g++
CC := $(CC_PREFIX)${TARGET_TOOLCHAIN_PREFIX}gcc
AR := $(CC_PREFIX)${TARGET_TOOLCHAIN_PREFIX}ar

MINIMAL_OBJS := $(addprefix $(OBJDIR), \
$(patsubst %.cc,%.o,$(patsubst %.c,%.o,$(MINIMAL_SRCS))))

LIB_OBJS := $(addprefix $(OBJDIR), \
$(patsubst %.cc,%.o,$(patsubst %.c,%.o,$(TF_LITE_CC_SRCS))))

TEST_OBJS := $(addprefix $(OBJDIR), \
$(patsubst %.cc,%.o,$(patsubst %.c,%.o,$(TEST_SRCS))))

BENCHMARK_OBJS := $(addprefix $(OBJDIR), \
$(patsubst %.cc,%.o,$(patsubst %.c,%.o,$(BENCHMARK_SRCS))))

#GEM5_OBJ := $(addprefix $(OBJDIR), \
#$(patsubst %.S,%.o, $(GEM5_SRCS)))

# For normal manually-created TensorFlow C++ source files.
$(OBJDIR)%.o: %.cc
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@
# For normal manually-created TensorFlow C++ source files.
$(OBJDIR)%.o: %.c
	@mkdir -p $(dir $@)
	$(CC) $(CCFLAGS) $(INCLUDES) -c $< -o $@
# For normal manually-created TensorFlow ASM source files.
$(OBJDIR)%.o: %.S
	@mkdir -p $(dir $@)
	$(CC) $(CCFLAGS) $(INCLUDES) -c $< -o $@
# The target that's compiled if there's no command-line arguments.
all: $(LIB_PATH) $(MINIMAL_BINARY)

test: $(LIB_PATH) $(TEST_OBJS) $(TEST_BINARY)

benchmark: $(LIB_PATH) $(BENCHMARK_BINARY)

# Hack for generating schema file bypassing flatbuffer parsing
tensorflow/contrib/lite/schema/schema_generated.h:
	@cp -u tensorflow/contrib/lite/schema/schema_generated.h.OPENSOURCE tensorflow/contrib/lite/schema/schema_generated.h

# Gathers together all the objects we've compiled into a single '.a' archive.
$(LIB_PATH): tensorflow/contrib/lite/schema/schema_generated.h $(LIB_OBJS)
	@mkdir -p $(dir $@)
	$(AR) $(ARFLAGS) $(LIB_PATH) $(LIB_OBJS)

$(MINIMAL_BINARY): $(MINIMAL_OBJS) $(LIB_PATH)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) \
	-o $(MINIMAL_BINARY) $(MINIMAL_OBJS)\
	$(LIBFLAGS) $(LIB_PATH) $(LDFLAGS) $(LIBS)

$(TEST_BINARY): $(BINDIR)% : $(OBJDIR)%.o $(LIB_PATH)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) \
	-o $@ $^ \
	$(LIBFLAGS) $(LIB_PATH) $(LDFLAGS) $(LIBS)


$(BENCHMARK_BINARY): $(BINDIR)% : $(OBJDIR)%.o $(LIB_PATH)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) \
	-o $@ $^ \
	$(LIBFLAGS) $(LIB_PATH) $(LDFLAGS) $(LIBS)

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

-include $(patsubst %,$(DEPDIR)/%.d,$(basename $(ALL_SRCS)))
