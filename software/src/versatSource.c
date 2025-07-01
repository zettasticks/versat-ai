#include "versat_ai.h"

#include "stdbool.h"
#include "stdint.h"

#include "iob_printf.h"

#include "versat_accel.h"

void *Versat_Add(void *inputA, void *inputB, void *output, int index,
                 AddInfo *info) {
  float *viewA = (float *)inputA;
  float *viewB = (float *)inputB;
  float *out = (float *)output;

  printf("Using Versat\n");

  printf("Addr: %x",&accelConfig->inputs_0);

  ReadDataBroadCasted_VRead(&accelConfig->inputs_0, inputA, 4, 2);
  ReadDataBroadCasted_VRead_2(&accelConfig->inputs_1, inputB, 4, 2);
  WriteDataBroadCasted_VWrite(&accelConfig->output, output, 4, 2);

  RunAccelerator(3);

  return output;
}
