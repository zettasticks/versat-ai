/*
 * SPDX-FileCopyrightText: 2026 IObundle, Lda
 *
 * SPDX-License-Identifier: MIT
 *
 * Py2HWSW Version 0.81.0 has generated this code
 * (https://github.com/IObundle/py2hwsw).
 */

#include "iob_regfileif_inverted_csrs.h"

// Base Address
static uint32_t base;
void iob_regfileif_inverted_csrs_init_baseaddr(uint32_t addr) { base = addr; }

// Core Setters and Getters
void iob_regfileif_inverted_csrs_set_start(uint8_t value) {
  iob_write(base + IOB_REGFILEIF_INVERTED_CSRS_START_ADDR,
            IOB_REGFILEIF_INVERTED_CSRS_START_W, value);
}

uint8_t iob_regfileif_inverted_csrs_get_start() {
  return iob_read(base + IOB_REGFILEIF_INVERTED_CSRS_START_ADDR,
                  IOB_REGFILEIF_INVERTED_CSRS_START_W);
}

void iob_regfileif_inverted_csrs_set_done(uint8_t value) {
  iob_write(base + IOB_REGFILEIF_INVERTED_CSRS_DONE_ADDR,
            IOB_REGFILEIF_INVERTED_CSRS_DONE_W, value);
}

uint32_t iob_regfileif_inverted_csrs_get_metamodel_addr() {
  return iob_read(base + IOB_REGFILEIF_INVERTED_CSRS_METAMODEL_ADDR_ADDR,
                  IOB_REGFILEIF_INVERTED_CSRS_METAMODEL_ADDR_W);
}

uint32_t iob_regfileif_inverted_csrs_get_output_addr() {
  return iob_read(base + IOB_REGFILEIF_INVERTED_CSRS_OUTPUT_ADDR_ADDR,
                  IOB_REGFILEIF_INVERTED_CSRS_OUTPUT_ADDR_W);
}

uint32_t iob_regfileif_inverted_csrs_get_temp_addr() {
  return iob_read(base + IOB_REGFILEIF_INVERTED_CSRS_TEMP_ADDR_ADDR,
                  IOB_REGFILEIF_INVERTED_CSRS_TEMP_ADDR_W);
}

uint32_t iob_regfileif_inverted_csrs_get_model_addr() {
  return iob_read(base + IOB_REGFILEIF_INVERTED_CSRS_MODEL_ADDR_ADDR,
                  IOB_REGFILEIF_INVERTED_CSRS_MODEL_ADDR_W);
}

uint32_t iob_regfileif_inverted_csrs_get_correctOutputs_addr() {
  return iob_read(base + IOB_REGFILEIF_INVERTED_CSRS_CORRECTOUTPUTS_ADDR_ADDR,
                  IOB_REGFILEIF_INVERTED_CSRS_CORRECTOUTPUTS_ADDR_W);
}

uint32_t iob_regfileif_inverted_csrs_get_inputsVector_addr() {
  return iob_read(base + IOB_REGFILEIF_INVERTED_CSRS_INPUTSVECTOR_ADDR_ADDR,
                  IOB_REGFILEIF_INVERTED_CSRS_INPUTSVECTOR_ADDR_W);
}

uint8_t iob_regfileif_inverted_csrs_get_rst() {
  return iob_read(base + IOB_REGFILEIF_INVERTED_CSRS_RST_ADDR,
                  IOB_REGFILEIF_INVERTED_CSRS_RST_W);
}

uint32_t iob_regfileif_inverted_csrs_get_version() {
  return iob_read(base + IOB_REGFILEIF_INVERTED_CSRS_VERSION_ADDR,
                  IOB_REGFILEIF_INVERTED_CSRS_VERSION_W);
}
