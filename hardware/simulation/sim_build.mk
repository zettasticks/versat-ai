# SPDX-FileCopyrightText: 2025 IObundle
#
# SPDX-License-Identifier: MIT

include auto_sim_build.mk

# Add iob-soc software as a build dependency
BUILD_DEPS+=versat_ai_bootrom.hex versat_ai_firmware.hex

ROOT_DIR :=../..
include $(ROOT_DIR)/software/sw_build.mk

ifeq ($(USE_ETHERNET),1)
VSRC+=./src/iob_eth_csrs_emb_verilator.c ./src/iob_eth_driver_tb.cpp
endif

VLT_SRC = ../../software/src/iob_uart_csrs.c

CONSOLE_CMD ?=rm -f soc2cnsl cnsl2soc; ../../scripts/console.py -L

GRAB_TIMEOUT ?= 3600

TBTYPE?=C
ifeq ($(VCD),1)
# Enable FST traces: 
# if TRACE_FST is defined in CPP, simulation outputs FST instead of VCD
VFLAGS+=--trace-fst
endif

CUSTOM_COVERAGE_FLAGS=cov_annotated
CUSTOM_COVERAGE_FLAGS+=-o versat_ai_coverage.rpt
