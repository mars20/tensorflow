# Settings for ARM-64 bit platforms.
ifeq ($(TARGET), aarch64)
  TARGET_ARCH := aarch4
  TARGET_TOOLCHAIN_PREFIX := aarch64-linux-gnu-
endif

