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

  AddressVArguments args0 = CompileVUnit_DataBroadCasted(inputA, 4, GetDim(l, d, 0),
                        GetDim(l, d, 1), GetDim(l, d, 2), GetDim(l, d, 3),
                        GetDim(l, d, 4));

  AddressVArguments args1 = CompileVUnit_DataBroadCasted(inputB, 4, GetDim(r, d, 0),
                          GetDim(r, d, 1), GetDim(r, d, 2), GetDim(r, d, 3),
                          GetDim(r, d, 4));

  int loopSize0 = CalculateLoopSize(args0);
  int loopSize1 = CalculateLoopSize(args1);

#if 0
  DataBroadCasted_VRead(&accelConfig->inputs_0, inputA, 4, GetDim(l, d, 0),
                        GetDim(l, d, 1), GetDim(l, d, 2), GetDim(l, d, 3),
                        GetDim(l, d, 4));
  DataBroadCasted_VRead(&accelConfig->inputs_1, inputB, 4, GetDim(r, d, 0),
                          GetDim(r, d, 1), GetDim(r, d, 2), GetDim(r, d, 3),
                          GetDim(r, d, 4));
#endif

  DataBroadCasted_VWrite(&accelConfig->output, output, 1, GetDim(o, d, 0),
                         GetDim(o, d, 1), GetDim(o, d, 2), GetDim(o, d, 3),
                         GetDim(o, d, 4));

  RunAccelerator(3);

  return output;
}
