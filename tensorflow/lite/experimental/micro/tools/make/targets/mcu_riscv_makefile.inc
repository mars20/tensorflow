# Settings for RISCV MCU toolchain.
ifeq ($(TARGET), riscv32_mcu)
  TARGET_ARCH := riscv32_mcu
  TARGET_TOOLCHAIN_PREFIX := riscv64-unknown-elf-
  
  PLATFORM_FLAGS = \
    -march=rv32imac \
    -mabi=ilp32 \
    -mcmodel=medany \
    -mexplicit-relocs \
    -fno-common \
    -falign-functions=4 \
    -fno-builtin-printf \
    -DTF_LITE_MCU_DEBUG_LOG \
    -DTF_LITE_RISCV_MCU
    
  CXXFLAGS += $(PLATFORM_FLAGS) \
    -fpermissive \
    -fno-exceptions \
    --std=gnu++11 
    
  CCFLAGS += $(PLATFORM_FLAGS) 
  
  LDFLAGS += \
   -T$(MAKEFILE_DIR)/downloads/sifive_fe310_lib/bsp/env/freedom-e300-hifive1/flash.lds \
   -L$(MAKEFILE_DIR)/downloads/sifive_fe310_lib/bsp/env \
   -nostartfiles \
   --specs=nano.specs
   
  BUILD_TYPE := micro
  
  INCLUDES += \
    -I$(MAKEFILE_DIR)/downloads/sifive_fe310_lib/bsp/include \
    -I$(MAKEFILE_DIR)/downloads/sifive_fe310_lib/bsp/drivers/ \
    -I$(MAKEFILE_DIR)/downloads/sifive_fe310_lib/bsp/env \
    -I$(MAKEFILE_DIR)/downloads/sifive_fe310_lib/bsp/env/freedom-e300-hifive1
  
  MICROLITE_CC_SRCS += \
    $(wildcard tensorflow/lite/experimental/micro/riscv32_mcu/*.cc)

  MICRO_SPEECH_TEST_SRCS += \
    $(wildcard $(MAKEFILE_DIR)/downloads/sifive_fe310_lib/bsp/libwrap/sys/*.c) \
    $(wildcard $(MAKEFILE_DIR)/downloads/sifive_fe310_lib/bsp/libwrap/sys/*.cc) \
    $(MAKEFILE_DIR)/downloads/sifive_fe310_lib/bsp/libwrap/misc/write_hex.c \
    $(MAKEFILE_DIR)/downloads/sifive_fe310_lib/bsp/libwrap/stdlib/malloc.c \
    $(MAKEFILE_DIR)/downloads/sifive_fe310_lib/bsp/env/start.S \
    $(MAKEFILE_DIR)/downloads/sifive_fe310_lib/bsp/env/entry.S \
    $(MAKEFILE_DIR)/downloads/sifive_fe310_lib/bsp/env/freedom-e300-hifive1/init.c

  LIBWRAP_SYMS := malloc free \
	open lseek read write fstat stat close link unlink \
	execve fork getpid kill wait \
	isatty times sbrk _exit puts

  LDFLAGS += $(foreach s,$(LIBWRAP_SYMS),-Wl,--wrap=$(s))
  LDFLAGS += $(foreach s,$(LIBWRAP_SYMS),-Wl,--wrap=_$(s))
  LDFLAGS += -L. -Wl,--start-group -lc -Wl,--end-group

# These are microcontroller-specific rules for converting the ELF output
# of the linker into a binary image that can be loaded directly.
OBJCOPY := $(TARGET_TOOLCHAIN_PREFIX)objcopy

$(BINDIR)/%.bin: $(BINDIR)/%
	@mkdir -p $(dir $@)
	$(OBJCOPY) $< $@ -O binary
endif