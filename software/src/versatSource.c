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

  DataBroadCasted_VRead(  &accelConfig->inputs_0, inputA,2,4);
  DataBroadCasted_VRead_2(&accelConfig->inputs_1, inputB,2,4);
  DataBroadCasted_VWrite( &accelConfig->output  , output,2,4);

#if 0
  DataBroadCasted_VRead(  &accelConfig->inputs_0, inputA,2,4,0,0,0,0);
  DataBroadCasted_VRead_2(&accelConfig->inputs_1, inputB,2,4,0,0,0,0);
  DataBroadCasted_VWrite( &accelConfig->output  , output,2,4,0,0,0,0);
#endif

  RunAccelerator(3);

  return output;
}
