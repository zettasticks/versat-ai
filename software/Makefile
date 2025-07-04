# SPDX-FileCopyrightText: 2025 IObundle
#
# SPDX-License-Identifier: MIT

# This makefile is used at build-time in $(BUILD_DIR)/software/Makefile
include ../config_build.mk

-include VerilatorMake.mk

#########################################
#            General settings           #
#########################################

PYTHON_DIR = ../scripts

# include local build segment
# all previously defined variables can be overwritten in this file
ifneq ($(wildcard sw_build.mk),)
include sw_build.mk
endif

#########################################
#           Embedded settings           #
#########################################

TEMPLATE_LDS ?=template.lds

MFLAGS=$(MFLAGS_BASE)$(MFLAG_M)$(MFLAG_C)
MFLAGS_BASE:=rv32i
ifeq ($(USE_MUL_DIV),1)
MFLAG_M=m
endif
ifeq ($(USE_COMPRESSED),1)
MFLAG_C=c
endif
ifeq ($(NO_HW_DRIVER),)
EXTRA_SRC+=src/hw_driver.c
endif

#default compiler settings
TOOLCHAIN_PREFIX ?=riscv64-unknown-elf-
CFLAGS ?=-Os -nostdlib -march=$(MFLAGS) -mabi=ilp32 --specs=nano.specs -Wcast-align=strict $(USER_CFLAGS)
LFLAGS ?= -Wl,-L,src,-Bstatic,-T,$(TEMPLATE_LDS),--strip-debug
LLIBS ?=-lgcc -lc -lnosys
INCLUDES ?=-I. -Isrc

#########################################
#         PC emulation settings         #
#########################################

# compiler flags
EMUL_CFLAGS ?=-std=gnu99 -g -DPC

CONSOLE_CMD ?=rm -f soc2cnsl cnsl2soc; ../scripts/console.py -L

EMUL_INCLUDES ?=-I. -Isrc

#########################################
#            Embedded targets           #
#########################################

build: $(UTARGETS)
%.elf: $(TEMPLATE_LDS) $(HDR) $(SRC) $(EXTRA_SRC)
	$(TOOLCHAIN_PREFIX)gcc -o $@ $(CFLAGS) $(LFLAGS) $(INCLUDES) $(SRC) $(EXTRA_SRC) $(LLIBS)
	$(TOOLCHAIN_PREFIX)objcopy -O binary $@ $(*F).bin

ifneq ($(BOARD),)
FPGA_TOOL:=$(shell find ../hardware/fpga -name $(BOARD) | cut -d"/" -f5)
endif

debug:
	@echo $(TEMPLATE_LDS)
	@echo $(FW_SRC)
	@echo $(BOOT_SRC)
	@echo $(HDR)

.PHONY: build debug

#########################################
#         PC emulation targets          #
#########################################

fw_emul: $(EMUL_HDR) $(EMUL_SRC) libaccel.a
	gcc -o $@ $(EMUL_CFLAGS) $(EMUL_INCLUDES) $(EMUL_SRC) -lm libaccel.a -lstdc++

build_emul: fw_emul
	@echo "build"

#board client command
BOARD_GRAB_CMD ?=../scripts/board_client.py grab 300

test.log: build_emul
	$(BOARD_GRAB_CMD) -s './fw_emul' -c '$(CONSOLE_CMD)'

run_emul: test.log
	test "$$(cat test.log)" = "Test passed!"

.PHONY: build_emul run_emul

#########################################
#            Simulation targets         #
#########################################

ifneq ($(wildcard ./src/$(NAME)_csrs.c),)
CSRS ?= ./src/$(NAME)_csrs.c
endif

TB_INCLUDES ?=-I./src

tb: ./src/iob_c_tb.c ./src/iob_core_tb.c $(CSRS)
	gcc $(TB_INCLUDES) -g -o $@ $^


#########################################
#            General targets            #
#########################################

clean:
	@rm -rf *.bin *.elf *.map *.hex tb
	@rm -rf fw_emul test.log soc2cnsl cnsl2soc
	@rm -rf *.txt iob_soc_conf.h

.PHONY: clean
