# Settings for RISCV-64 bit platforms.
ifeq ($(TARGET), riscv64)
  TARGET_ARCH := riscv64
  TARGET_TOOLCHAIN_PREFIX := riscv64-unknown-elf-
  CXXFLAGS += --std=gnu++11
endif

