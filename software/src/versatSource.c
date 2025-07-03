#include "versat_ai.h"

#include "stdbool.h"
#include "stdint.h"

#include "iob_printf.h"

#include "versat_accel.h"

#define MAX(A, B) (((A > B) ? (A) : (B)))

void clear_cache();

inline int64_t GetDim(int64_t *dimArray, int dimSize, int index) {
  if (index < dimSize) {
    return MAX(dimArray[index], 1);
  }

  return 1;
}

void *Versat_Add(void *inputA, void *inputB, void *output, int index,
                 AddInfo *info) {
  float *viewA = (float *)inputA;
  float *viewB = (float *)inputB;
  float *out = (float *)output;

  printf("Using Versat\n");

  printf("%p\n", &accelConfig->inputs_0);

  int64_t *l = info->firstInputDim;
  int64_t *r = info->secondInputDim;
  int64_t *o = info->broadCastedShape;
  int d = info->maxDims;

  //- LEFT HERE - Somewhere along the interconnection, the rdata that goes from
  //the memory to Versat becomes different. Probably something to do with the
  //inteconnect.

  DataBroadCasted_VRead(&accelConfig->inputs_0, inputA, GetDim(l, d, 0),
                        GetDim(l, d, 1), GetDim(l, d, 2), GetDim(l, d, 3),
                        GetDim(l, d, 4), GetDim(l, d, 5));
  DataBroadCasted_VRead_2(&accelConfig->inputs_1, inputB, GetDim(r, d, 0),
                          GetDim(r, d, 1), GetDim(r, d, 2), GetDim(r, d, 3),
                          GetDim(r, d, 4), GetDim(r, d, 5));
  DataBroadCasted_VWrite(&accelConfig->output, output, GetDim(o, d, 0),
                         GetDim(o, d, 1), GetDim(o, d, 2), GetDim(o, d, 3),
                         GetDim(o, d, 4), GetDim(o, d, 5));

  RunAccelerator(3);

  return output;
}
