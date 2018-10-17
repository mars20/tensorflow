# Settings for RISCV-64 bit platforms.
ifeq ($(TARGET), riscv64EXT)
  TARGET_ARCH := riscv64EXT
  TARGET_TOOLCHAIN_PREFIX := riscv64-unknown-elf-
  CXXFLAGS += -march=rv64gv -DRISCV 
endif