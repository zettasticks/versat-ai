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

inline int64_t GetSize(int64_t *dimArray, int dimSize, int index){
#if 0
  if (index < dimSize) {
    return MAX(dimArray[index], 1) - 1;
  }
#endif
  if (index < dimSize) {
    if(dimArray[index] > 1){
      return 1;
    }
  }

  return 0;
}

void *Versat_Add(void *inputA, void *inputB, void *output, int index,
                 AddInfo *info) {
  float *viewA = (float *)inputA;
  float *viewB = (float *)inputB;
  float *out = (float *)output;

  printf("Using Versat\n");

  int64_t *l = info->firstInputDim;
  int64_t *r = info->secondInputDim;
  int64_t *o = info->broadCastedShape;
  int d = info->maxDims;

  DataBroadCasted_VRead(&accelConfig->inputs_0,inputA, GetSize(l, d, 0), GetDim(o, d, 0),
                                                       GetSize(l, d, 1), GetDim(o, d, 1),GetDim(l,d,1),
                                                       GetSize(l, d, 2), GetDim(o, d, 2),GetDim(l,d,2),
                                                       GetSize(l, d, 3), GetDim(o, d, 3),GetDim(l,d,3),
                                                       GetSize(l, d, 4), GetDim(o, d, 4),GetDim(l,d,4),
                                                       GetSize(l, d, 5), GetDim(o, d, 5),GetDim(l,d,5));

  DataBroadCasted_VRead(&accelConfig->inputs_1,inputB, GetSize(r, d, 0), GetDim(o, d, 0),
                                                       GetSize(r, d, 1), GetDim(o, d, 1),GetDim(r,d,1),
                                                       GetSize(r, d, 2), GetDim(o, d, 2),GetDim(r,d,2),
                                                       GetSize(r, d, 3), GetDim(o, d, 3),GetDim(r,d,3),
                                                       GetSize(r, d, 4), GetDim(o, d, 4),GetDim(r,d,4),
                                                       GetSize(r, d, 5), GetDim(o, d, 5),GetDim(r,d,5));

  DataSimple_VWrite(&accelConfig->output, output, GetDim(o, d, 0),
                                                  GetDim(o, d, 1),
                                                  GetDim(o, d, 2), 
                                                  GetDim(o, d, 3),
                                                  GetDim(o, d, 4),
                                                  GetDim(o, d, 5));

  RunAccelerator(3);

  return output;
}
