#include "iob_versat_csrs.h"

// Base Address
static uint32_t base;
void iob_versat_csrs_init_baseaddr(uint32_t addr) { base = addr; }

// Core Setters and Getters
uint32_t iob_versat_csrs_get_interface(int addr) {
  return iob_read(base + IOB_VERSAT_CSRS_INTERFACE_ADDR,
                  IOB_VERSAT_CSRS_INTERFACE_W);
}

uint16_t iob_versat_csrs_get_version() {
  return iob_read(base + IOB_VERSAT_CSRS_VERSION_ADDR,
                  IOB_VERSAT_CSRS_VERSION_W);
}
