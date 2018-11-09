# Settings for ARM-64 bit platforms.
ifeq ($(TARGET), aarch64)
  TARGET_ARCH := aarch64
  TARGET_TOOLCHAIN_PREFIX := aarch64-linux-gnu-
  CXXFLAGS += -static -pthread -Wl,--whole-archive -lpthread -Wl,--no-whole-archive -DUSE_NEON -DARM_GEM5 -mtune=cortex-a53
endif
ifeq ($(TARGET), aarch32)
  TARGET_ARCH := aarch32
  TARGET_TOOLCHAIN_PREFIX := arm-linux-gnueabi-
  CXXFLAGS += -mfpu=neon -mfloat-abi=softfp
  LIBS += -latomic

endif

