# Settings for ARM-64 bit platforms.
ifeq ($(TARGET), aarch64)
  TARGET_ARCH := aarch64
  TARGET_TOOLCHAIN_PREFIX := aarch64-linux-gnu-
  CXXFLAGS += -static -mtune=cortex-a53 -DARM_GEM5 
endif

