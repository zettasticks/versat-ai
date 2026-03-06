#include "versat_private.h"

#define MAX(A, B) ((A) > (B) ? (A) : (B))
#define MIN(A, B) ((A) < (B) ? (A) : (B))

//#define LOW_OUTPUT

// ======================================
// Global stuff (versat side)

uint64_t Versat_DefaultMeasureTime() { return 0; }
void Versat_DefaultClearCache(void *ptr, size_t size) {}

MeasureTimeFunction versat_time = Versat_DefaultMeasureTime;
ClearCache versat_clearCache = Versat_DefaultClearCache;

MeasureTimeFunction
Versat_SetTimeMeasurementFunction(MeasureTimeFunction func) {
  MeasureTimeFunction old = versat_time;
  versat_time = func;
  return old;
}

ClearCache Versat_SetClearCache(ClearCache func) {
  ClearCache old = versat_clearCache;
  versat_clearCache = func;
  return old;
}

void Assert_(bool cond, const char *msg, int line, const char *file) {
  if (!cond) {
    versat_printf("Assertion failed!\n");
    versat_printf("In file %s:%d\n", file, line);
    versat_printf("%s", msg);
  }
}

#define Assert(COND, MSG) Assert_(COND, MSG, __LINE__, __FILE__)

// ======================================
// Dimensions

Dimensions CreateDimensions(int64_t *dims, int numberDims) {
  Dimensions res = {};
  res.size = numberDims;
  for (int i = 0; i < numberDims; i++) {
    res.data[i] = dims[i];
  }
  return res;
}

void Dimensions_PrependInPlace(Dimensions *dim, int value) {
  Assert(dim->size + 1 <= MAX_DIMS, "MAX_DIMS overflow");
  for (int i = 0; i < dim->size; i++) {
    dim->data[i + 1] = dim->data[i];
  }
  dim->data[0] = value;
  dim->size += 1;
}

void Dimensions_AppendInPlace(Dimensions *dim, int value) {
  Assert(dim->size + 1 <= MAX_DIMS, "MAX_DIMS overflow");
  dim->data[dim->size] = value;
  dim->size += 1;
}

Dimensions Dimensions_Cut_GetLeft(Dimensions dim, int amount) {
  Dimensions res = {};

  if (amount == 0) {
    res.data[0] = 1;
    res.size = 1;
    return res;
  }

  int size = MIN(dim.size, amount);

  for (int i = 0; i < size; i++) {
    res.data[i] = dim.data[i];
  }
  res.size = size;

  return res;
}

Dimensions Dimensions_Cut_GetRight(Dimensions dim, int amount) {
  Dimensions res = {};

  if (amount == dim.size) {
    res.data[0] = 1;
    res.size = 1;
    return res;
  }

  int size = MAX(dim.size - amount, 0);
  for (int i = 0; i < size; i++) {
    res.data[i] = dim.data[amount + i];
  }
  res.size = size;

  return res;
}

int Dimensions_TotalSize(Dimensions dim) {
  int size = 1;
  for (int i = 0; i < dim.size; i++) {
    size *= dim.data[i];
  }
  return size;
}

// ======================================
// Address

// Proper dims are the dims used to calculate an index.
// Iteration dims are the iterations.
AddressGen StartAddress(int64_t *iterationDims, int64_t *properDims,
                        int numberDims) {
  AddressGen gen = {};

  for (int i = 0; i < numberDims; i++) {
    gen.iterationDims[i] = iterationDims[i];
    gen.properDims[i] = properDims[i];
  }
  gen.numberDims = numberDims;

  return gen;
}

AddressGen StartAddressFromDims(Dimensions dims, int iterDims) {
  AddressGen gen = {};

  for (int i = 0; i < dims.size; i++) {
    gen.properDims[i] = dims.data[i];
    gen.iterationDims[i] = dims.data[i];

    if (i >= iterDims) {
      gen.iterationDims[i] = 1;
    }
  }
  gen.numberDims = dims.size;

  return gen;
}

int Address_GetDim(AddressGen *gen, int index) {
  Assert(index < gen->numberDims, "Index greater than dimensions of Address");
  return gen->addressVars[index];
}

void Address_Print(AddressGen *gen) {
  for (int i = 0; i < gen->numberDims; i++) {
    if (i != 0) {
      versat_printf(" x ");
    }
    versat_printf("%d", gen->addressVars[i]);
  }

  versat_printf(" [");
  for (int i = 0; i < gen->numberDims; i++) {
    if (i != 0) {
      versat_printf(" x ");
    }
    versat_printf("%ld", gen->iterationDims[i]);
  }
  versat_printf("]\n");
}

int Address_GetValue(AddressGen *gen) {
  int address = 0;
  for (int i = 0; i < gen->numberDims; i++) {
    int index = gen->addressVars[i] + gen->offsetAddressVars[i];

    if (index >= gen->properDims[i]) {
      index = 0;
    }

    if (i > 0) {
      address *= gen->properDims[i];
    }
    address += index;
  }

  return address;
}

bool Address_IsValid(AddressGen *gen) {
  // Because we allow out of order advances, we need to check every
  for (int i = 0; i < gen->numberDims; i++) {
    if (gen->addressVars[i] >= gen->iterationDims[i]) {
      return false;
    }
  }

  for (int i = 0; i < gen->numberDims; i++) {
    if (gen->addressVars[i] >= gen->iterationDims[i]) {
      continue;
    } else {
      return true;
    }
  }
  return false;
}

void Address_Advance(AddressGen *gen) {
  if (gen->addressVars[0] >= gen->iterationDims[0]) {
    return;
  }

  for (int i = gen->numberDims - 1; i >= 0; i--) {
    if (i != 0 && gen->addressVars[i] + 1 >= gen->iterationDims[i]) {
      gen->addressVars[i] = 0;
      continue;
    } else {
      gen->addressVars[i] += 1;
      return;
    }
  }
}

void Address_AdvanceAxis(AddressGen *gen, int axisToAdvance) {
  // Any negative axis just puts the address gen into an invalid state
  if (axisToAdvance < 0) {
    gen->addressVars[0] = gen->iterationDims[0] + 1;
    return;
  }

  if (gen->addressVars[0] >= gen->iterationDims[0]) {
    return;
  }

  for (int i = axisToAdvance; i >= 0; i--) {
    if (i != 0 && gen->addressVars[i] + 1 >= gen->iterationDims[i]) {
      gen->addressVars[i] = 0;
      continue;
    } else {
      gen->addressVars[i] += 1;
      return;
    }
  }
}

AddressGen Address_Map(AddressGen *in, int64_t *biggerDim, int *stride) {
  AddressGen gen = *in;

  for (int i = 0; i < in->numberDims; i++) {
    gen.addressVars[i] *= stride[i];
    gen.iterationDims[i] = biggerDim[i];
    gen.properDims[i] = biggerDim[i];
  }

  return gen;
}

AddressGen Address_Map2(AddressGen *in, int64_t *biggerDim, int *stride,
                        int *offset) {
  AddressGen gen = *in;

  for (int i = 0; i < in->numberDims; i++) {
    gen.addressVars[i] *= stride[i];
    gen.addressVars[i] -= offset[i];
    gen.iterationDims[i] = biggerDim[i];
    gen.properDims[i] = biggerDim[i];
  }

  return gen;
}

void Address_Restart(AddressGen *gen) {
  for (int i = 0; i < gen->numberDims; i++) {
    gen->addressVars[i] = 0;
  }
}

// ======================================
// KernelGen

// KernelDims are the bounds of the dimensions that the KernelGen iterates over
// Example, if we have a layer of dim A,B,C,D and kernelSize of 2, then
// the kernel only iterates over the C and D dimensions, never A or B.
// kernelDims has kernelSize size and defines the boundary of the iteration
KernelGen StartKernel(AddressGen *address, int *kernelDims, int kernelSize) {
  KernelGen gen = {};
  // gen.address = address;

  int nonKernelDims = address->numberDims - kernelSize;

  for (int i = 0; i < nonKernelDims; i++) {
    gen.kernelDims[i] = 1;
  }
  gen.numberDims = address->numberDims;

  for (int i = 0; i < MAX_DIMS; i++) {
    gen.kernelDilations[i] = 1;
  }

  for (int i = 0; i < kernelSize; i++) {
    gen.kernelDims[nonKernelDims + i] = kernelDims[i];
  }

  for (int i = 0; i < address->numberDims; i++) {
    gen.addressGenVars[i] = address->addressVars[i];
  }
  for (int i = 0; i < address->numberDims; i++) {
    gen.addressIterDims[i] = address->iterationDims[i];
  }
  for (int i = 0; i < address->numberDims; i++) {
    gen.addressProperDims[i] = address->properDims[i];
  }

  return gen;
}

KernelGen StartKernel_IterateOneDimOnly(AddressGen *address, int dimToIterate,
                                        int start, int end) {
  int dims[] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};

  KernelGen gen = StartKernel(address, dims, address->numberDims);

  gen.addressGenVars[dimToIterate] = 0;
  gen.kernelDims[dimToIterate] = end;
  gen.kernelVars[dimToIterate] = start;

  return gen;
}

void Kernel_PrintShort(KernelGen *gen) {
  versat_printf(
      "Kernel is gonna iterate the base tensor in the following coordinates:");

  versat_printf(" [");
  for (int i = 0; i < gen->numberDims; i++) {
    if (i != 0) {
      versat_printf(" x ");
    }
    versat_printf("%ld - %ld", gen->addressGenVars[i],
                  gen->addressGenVars[i] + gen->kernelDims[i]);
  }
  versat_printf("]\n");
}

void Kernel_Print(KernelGen *gen) {
  for (int i = 0; i < gen->numberDims; i++) {
    if (i != 0) {
      versat_printf(" x ");
    }
    versat_printf("%d", gen->kernelVars[i]);
  }

  versat_printf(" [");
  for (int i = 0; i < gen->numberDims; i++) {
    if (i != 0) {
      versat_printf(" x ");
    }
    versat_printf("%ld", gen->kernelDims[i]);
  }
  versat_printf("]");

  versat_printf(" [");
  for (int i = 0; i < gen->numberDims; i++) {
    if (i != 0) {
      versat_printf(" x ");
    }
    versat_printf("%ld - %ld", gen->addressGenVars[i],
                  gen->addressGenVars[i] + gen->kernelDims[i]);
  }
  versat_printf("]\n");
}

int Kernel_GetValue(KernelGen *gen) {
  int properVars[MAX_DIMS];

  for (int i = 0; i < gen->numberDims; i++) {
    properVars[i] =
        gen->kernelVars[i] * gen->kernelDilations[i] + gen->addressGenVars[i];
  }

  int address = 0;
  for (int i = 0; i < gen->numberDims; i++) {
    int index = properVars[i];

    if (index >= gen->addressIterDims[i]) {
      index = 0;
    }

    if (i > 0) {
      address *= gen->addressIterDims[i];
    }
    address += index;
  }

  return address;
}

bool Kernel_IsValid(KernelGen *gen) {
  // Is this the only thing that we need?
  if (gen->kernelVars[0] >= gen->kernelDims[0]) {
    return false;
  }

  for (int i = 0; i < gen->numberDims; i++) {
    if (gen->kernelVars[i] >= gen->kernelDims[i]) {
      continue;
    } else {
      return true;
    }
  }
  return false;
}

bool Kernel_IsInsidePad(KernelGen *gen) {
  int properVars[MAX_DIMS];

  for (int i = 0; i < gen->numberDims; i++) {
    properVars[i] =
        gen->kernelVars[i] * gen->kernelDilations[i] + gen->addressGenVars[i];
  }

  for (int i = 0; i < gen->numberDims; i++) {
    if (properVars[i] < 0 || properVars[i] >= gen->addressProperDims[i]) {
      // versat_printf("Bad: %d %d
      // %d\n",i,properVars[i],gen->addressProperDims[i]);
      return true;
    }
  }

  return false;
}

void Kernel_Advance(KernelGen *gen) {
  if (gen->kernelVars[0] >= gen->kernelDims[0]) {
    return;
  }

  for (int i = gen->numberDims - 1; i >= 0; i--) {
    if (i != 0 && gen->kernelVars[i] + 1 >= gen->kernelDims[i]) {
      gen->kernelVars[i] = 0;
      continue;
    } else {
      gen->kernelVars[i] += 1;
      return;
    }
  }
}

// ======================================
// Misc

int64_t CalculateSizeOfDim(int64_t *dim, int dims) {
  int64_t size = 1;
  for (int i = 0; i < dims; i++) {
    size *= dim[i];
  }

  return size;
}

static inline float absf(float a) {
  if (a < 0.0f) {
    return -a;
  }
  return a;
}

void AssertAlmostEqual(void *toTest, void *correctValues, int index,
                       LayerInfo *info) {
  float *test = (float *)toTest;
  float *correct = (float *)correctValues;

  size_t outputSize = info->outputSize / sizeof(float);

#ifndef LOW_OUTPUT
  // versat_printf("Gonna check output of layer: %d\n", index);
#endif

  if (outputSize == 0) {
    versat_printf(
        "Error, AssertAlmostEqual with output size of 0. Should not be "
        "possible\n");
    return;
  }

  int maxIncorrect = 10;
  bool printOk = true;

  // Make sure that cache is not affecting the verification process
  // TODO: Proper boundaries
  versat_clearCache(NULL, 0);

  int incorrectFound = 0;
  for (int i = 0; i < outputSize; i++) {
    if (absf(correct[i] - test[i]) > 0.001f) {
      if (incorrectFound == 0) {
        versat_printf("\n");
        versat_printf("[%s] (Layer %d) FAIL:\n", info->typeName, index);
      }
      versat_printf("  Index: %4d Different values %.4f %.4f\n", i, correct[i],
                    test[i]);
      if (i > 0) {
        versat_printf("    PreviousValue: %.4f\n", test[i - 1]);
      }
      incorrectFound += 1;
    }

    if (incorrectFound >= maxIncorrect) {
      versat_printf("More than %d incorrect found, quitting early\n",
                    maxIncorrect);
      versat_printf("\n");
      break;
    }
  }

#ifndef LOW_OUTPUT
  if (printOk && incorrectFound == 0) {
    versat_printf("[%30s] (Layer %4d) - OK\n", info->typeName, index);
  }
#endif
}

// Based on quake fast inverse square root function.
float my_invsqrt(float number) {
  long i;
  float x2, y;
  const float threehalfs = 1.5F;

  x2 = number * 0.5F;
  y = number;
  i = *(long *)&y;
  i = 0x5f3759df - (i >> 1);
  y = *(float *)&i;
  y = y * (threehalfs - (x2 * y * y));
  y = y * (threehalfs - (x2 * y * y));

  return y;
}

// ======================================
// Extra Info

void ExtraInfo_Print(ExtraInfo e) {
  versat_printf("ExtraInfo:\n");
  versat_printf("strideW: %d\n", e.strideW);
  versat_printf("strideH: %d\n", e.strideH);
  versat_printf("kernelW: %d\n", e.kernelW);
  versat_printf("kernelH: %d\n", e.kernelH);
  versat_printf("inputImageW: %d\n", e.inputImageW);
  versat_printf("inputImageH: %d\n", e.inputImageH);
  versat_printf("inputImageC: %d\n", e.inputImageC);
  versat_printf("outputImageW: %d\n", e.outputImageW);
  versat_printf("outputImageH: %d\n", e.outputImageH);
  versat_printf("outputImageC: %d\n", e.outputImageC);
  versat_printf("leftPadW: %d\n", e.leftPadW);
  versat_printf("leftPadH: %d\n", e.leftPadH);
  versat_printf("rightPadW: %d\n", e.rightPadW);
  versat_printf("rightPadH: %d\n", e.rightPadH);
  versat_printf("padW: %d\n", e.padW);
  versat_printf("padH: %d\n", e.padH);
  versat_printf("\n");
}

ExtraInfo CalculateExtraInfo_MaxPool(MaxPoolInfo *info) {
  ExtraInfo res = {};

  res.strideW = info->strideDims[1];
  res.strideH = info->strideDims[0];

  res.kernelW = info->kernelDims[1];
  res.kernelH = info->kernelDims[0];

  res.inputImageW = info->inputDims[3];
  res.inputImageH = info->inputDims[2];
  res.inputImageC = info->inputDims[1];

  res.outputImageC = info->outputDims[1];
  res.outputImageH = info->outputDims[2];
  res.outputImageW = info->outputDims[3];

  if (info->padding == PaddingType_NOTSET) {
    // TODO: Need a better way of handling errors in this layer, I think.
    if (info->padsSize != 4) {
      versat_printf("ERROR, pads size is not expected");
      return (ExtraInfo){};
    }

    res.leftPadW = info->padsDims[1];
    res.leftPadH = info->padsDims[0];

    res.rightPadW = info->padsDims[3];
    res.rightPadH = info->padsDims[2];

    res.padW = info->padsDims[1] + info->padsDims[3];
    res.padH = info->padsDims[0] + info->padsDims[2];
  } else if (info->padding == PaddingType_SAME_LOWER ||
             info->padding == PaddingType_SAME_UPPER) {
    res.padW = MAX(0, (res.outputImageW - 1) * res.strideW + res.kernelW -
                          res.inputImageW);
    res.padH = MAX(0, (res.outputImageH - 1) * res.strideH + res.kernelH -
                          res.inputImageH);

    int halfW = res.padW / 2;
    int halfH = res.padH / 2;

    res.leftPadW = halfW;
    res.rightPadW = halfW;
    res.leftPadH = halfH;
    res.rightPadH = halfH;

    if (res.padW % 2 == 1) {
      if (info->padding == PaddingType_SAME_LOWER) {
        res.leftPadW += 1;
      } else {
        res.rightPadW += 1;
      }
    }

    if (res.padH % 2 == 1) {
      if (info->padding == PaddingType_SAME_LOWER) {
        res.leftPadH += 1;
      } else {
        res.rightPadH += 1;
      }
    }
  }

  return res;
}

ExtraInfo CalculateExtraInfo_AveragePool(AveragePoolInfo *info) {
  ExtraInfo res = {};

  res.strideW = info->strideDims[1];
  res.strideH = info->strideDims[0];

  res.kernelW = info->kernelDims[1];
  res.kernelH = info->kernelDims[0];

  res.inputImageW = info->inputDims[3];
  res.inputImageH = info->inputDims[2];
  res.inputImageC = info->inputDims[1];

  res.outputImageC = info->outputDims[1];
  res.outputImageH = info->outputDims[2];
  res.outputImageW = info->outputDims[3];

  if (info->padding == PaddingType_NOTSET) {
    // TODO: Need a better way of handling errors in this layer, I think.

    if (info->padsSize != 4) {
      versat_printf("ERROR, pads size is not expected");
      return (ExtraInfo){};
    }

    res.leftPadW = info->padsDims[1];
    res.leftPadH = info->padsDims[0];

    res.rightPadW = info->padsDims[3];
    res.rightPadH = info->padsDims[2];

    res.padW = info->padsDims[1] + info->padsDims[3];
    res.padH = info->padsDims[0] + info->padsDims[2];
  } else if (info->padding == PaddingType_SAME_LOWER ||
             info->padding == PaddingType_SAME_UPPER) {
    res.padW = MAX(0, (res.outputImageW - 1) * res.strideW + res.kernelW -
                          res.inputImageW);
    res.padH = MAX(0, (res.outputImageH - 1) * res.strideH + res.kernelH -
                          res.inputImageH);

    int halfW = res.padW / 2;
    int halfH = res.padH / 2;

    res.leftPadW = halfW;
    res.rightPadW = halfW;
    res.leftPadH = halfH;
    res.rightPadH = halfH;

    if (res.padW % 2 == 1) {
      if (info->padding == PaddingType_SAME_LOWER) {
        res.leftPadW += 1;
      } else {
        res.rightPadW += 1;
      }
    }

    if (res.padH % 2 == 1) {
      if (info->padding == PaddingType_SAME_LOWER) {
        res.leftPadH += 1;
      } else {
        res.rightPadH += 1;
      }
    }
  }

  return res;
}

ExtraInfo CalculateExtraInfo_Conv(ConvInfo *info) {
  ExtraInfo res = {};

  res.strideW = info->strideDims[1];
  res.strideH = info->strideDims[0];

  res.kernelW = info->kernelDims[1];
  res.kernelH = info->kernelDims[0];

  res.inputImageW = info->inputDims[3];
  res.inputImageH = info->inputDims[2];
  res.inputImageC = info->inputDims[1];

  res.outputImageC = info->outputDims[1];
  res.outputImageH = info->outputDims[2];
  res.outputImageW = info->outputDims[3];

  if (info->padding == PaddingType_NOTSET) {
    // TODO: Need a better way of handling errors in this layer, I think.
    if (info->padsSize != 4) {
      versat_printf("ERROR, pads size is not expected");
      return (ExtraInfo){};
    }

    res.leftPadW = info->padsDims[1];
    res.leftPadH = info->padsDims[0];

    res.rightPadW = info->padsDims[3];
    res.rightPadH = info->padsDims[2];

    res.padW = info->padsDims[1] + info->padsDims[3];
    res.padH = info->padsDims[0] + info->padsDims[2];
  } else if (info->padding == PaddingType_SAME_LOWER ||
             info->padding == PaddingType_SAME_UPPER) {
    res.padW = MAX(0, (res.outputImageW - 1) * res.strideW + res.kernelW -
                          res.inputImageW);
    res.padH = MAX(0, (res.outputImageH - 1) * res.strideH + res.kernelH -
                          res.inputImageH);

    int halfW = res.padW / 2;
    int halfH = res.padH / 2;

    res.leftPadW = halfW;
    res.rightPadW = halfW;
    res.leftPadH = halfH;
    res.rightPadH = halfH;

    if (res.padW % 2 == 1) {
      if (info->padding == PaddingType_SAME_LOWER) {
        res.leftPadW += 1;
      } else {
        res.rightPadW += 1;
      }
    }

    if (res.padH % 2 == 1) {
      if (info->padding == PaddingType_SAME_LOWER) {
        res.leftPadH += 1;
      } else {
        res.rightPadH += 1;
      }
    }
  }

  return res;
}
// ======================================
// WindowGen

WindowGen StartWindowGen(ExtraInfo *info, bool iterateC, bool isNCHW) {
  WindowGen res = {};
  res.info = info;
  res.iterateC = iterateC;
  res.isNCHW = isNCHW;
  res.advanceC = 1;
  return res;
}

WindowGen StartAdvancedWindowGen(ExtraInfo *info, bool iterateC, bool isNCHW,
                                 int cMaxAdvance) {
  WindowGen res = {};
  res.info = info;
  res.iterateC = iterateC;
  res.isNCHW = isNCHW;
  res.advanceC = cMaxAdvance;
  return res;
}

void AdvancedWindow_Print(AdvancedWindow window) {
  bool printedOnce = false;
  if (window.padding & PaddingRegion_TOP) {
    versat_printf("Pad_TOP");
    printedOnce = true;
  }
  if (window.padding & PaddingRegion_BOTTOM) {
    if (printedOnce) {
      versat_printf(" | ");
    }
    versat_printf("Pad_BOTTOM");
    printedOnce = true;
  }
  if (window.padding & PaddingRegion_LEFT) {
    if (printedOnce) {
      versat_printf(" | ");
    }
    versat_printf("Pad_LEFT");
    printedOnce = true;
  }
  if (window.padding & PaddingRegion_RIGHT) {
    if (printedOnce) {
      versat_printf(" | ");
    }
    versat_printf("Pad_RIGHT");
    printedOnce = true;
  }

  versat_printf("\n");

  versat_printf("Output pos: X:%d,Y:%d (C:%d)\n", window.outputX,
                window.outputY, window.outputC);
  versat_printf("Input pos: (%d,%d)\n", window.inputX, window.inputY);
  versat_printf("WindowSize (Out view): %d %d %d\n", window.outputSizeC,
                window.outputH, window.outputW);
  versat_printf("KernelSizeAndOffset: %d:%d - %d:%d\n", window.actualKernelW,
                window.kernelStartW, window.actualKernelH, window.kernelStartH);
}

AdvancedWindow WindowGen_Get(WindowGen *gen) {
  AdvancedWindow res = {};

  res.outputX = gen->currentOutputX;
  res.outputY = gen->currentOutputY;
  res.outputC = gen->currentOutputC;

  res.startC = gen->currentOutputC;
  res.inputX = gen->currentOutputX * gen->info->strideW;
  res.inputY = gen->currentOutputY * gen->info->strideH;

  // Currently we assume a window size of 1, although need to add the better
  // logic to suport more windows and improve performance.

  // The only thing that we need to care about is the windows that are near
  // padding regions the fact that the accelerator must contain enough memory to
  // support a window and that we must make sure that the height of the window
  // is stable. ( So that we iterate over all the pixels correctly).
  res.outputW = 1;
  res.outputH = 1;

  // For now, just like the rest of the window, we only advance a single output
  // channel
  res.outputSizeC = gen->advanceC;

  if (res.outputSizeC + res.outputC >= gen->info->outputImageC) {
    res.outputSizeC = gen->info->outputImageC - res.outputC;
    if (res.outputSizeC <= 0) {
      versat_printf("ERROR, CANNOT HAVE OUTPUT SIZE LOWER OR EQUAL TO 0: %d",
                    res.outputSizeC);
    }
  }

  // By default, input equals kernel size
  res.actualKernelW = gen->info->kernelW;
  res.actualKernelH = gen->info->kernelH;

  res.inputX -= gen->info->leftPadW;
  res.inputY -= gen->info->leftPadH;

  // TODO: For the cases without padding, we can support bigger
  //       windows. We mainly want to center the logic around
  //       how much internal memory the accelerator supports (limiting factor
  //       for window size) and of course the boundaries between padding, since
  //       we cannot process different padding boundaries in the same run.

  // NOTE: Any amount of padding basically shifts the kernel and changes the
  // input window size.
  //       Difference between left and right padding is wether we change the
  //       start or not. The size of the kernel always changes.

  // This logic only works if we make sure that we can have a one by one window
  // size at the extreme points
  if (res.inputX < 0) {
    int offset = -res.inputX;
    res.actualKernelW -= offset;
    res.kernelStartW = offset;
    res.padding |= PaddingRegion_LEFT;
    res.inputX = 0;
  }
  if (res.inputX + res.actualKernelW > gen->info->inputImageW) {
    int offset = (res.inputX + res.actualKernelW) - gen->info->inputImageW;
    res.actualKernelW -= offset;
    res.padding |= PaddingRegion_RIGHT;
  }

  if (res.inputY < 0) {
    int offset = -res.inputY;
    res.actualKernelH -= offset;
    res.kernelStartH = offset;
    res.padding |= PaddingRegion_TOP;
    res.inputY = 0;
  }
  if (res.inputY + res.actualKernelH > gen->info->inputImageH) {
    int offset = (res.inputY + res.actualKernelH) - gen->info->inputImageH;
    res.actualKernelH -= offset;
    res.padding |= PaddingRegion_BOTTOM;
  }

  // In this case we are inside the entirity of the padding section, which means
  // that we do not actually want to do anything. It should just be zero, right?
  if (res.actualKernelH <= 0 || res.actualKernelW <= 0) {
    res.entireWindowInsidePadding = true;
  }

  return res;
}

void WindowGen_Advance(WindowGen *gen) {
  AdvancedWindow window = WindowGen_Get(gen);

  if (gen->iterateC) {
    if (gen->isNCHW) {
      gen->currentOutputX += window.outputW;
      if (gen->currentOutputX >= gen->info->outputImageW) {
        gen->currentOutputX = 0;
        gen->currentOutputY += window.outputH;
      }

      if (gen->currentOutputY >= gen->info->outputImageH) {
        gen->currentOutputY = 0;
        gen->currentOutputC += window.outputSizeC;
      }

      if (gen->currentOutputC >= gen->info->outputImageC) {
        gen->currentOutputC = -1;
        gen->currentOutputX = -1;
        gen->currentOutputY = -1;
      }
    } else {
      // NHWC
      gen->currentOutputC += window.outputSizeC;
      if (gen->currentOutputC >= gen->info->outputImageC) {
        gen->currentOutputC = 0;
        gen->currentOutputX += window.outputW;
      }

      if (gen->currentOutputX >= gen->info->outputImageW) {
        gen->currentOutputX = 0;
        gen->currentOutputY += window.outputH;
      }

      if (gen->currentOutputY >= gen->info->outputImageH) {
        gen->currentOutputC = -1;
        gen->currentOutputX = -1;
        gen->currentOutputY = -1;
      }
    }
  } else {
    gen->currentOutputX += window.outputW;
    if (gen->currentOutputX >= gen->info->outputImageW) {
      gen->currentOutputX = 0;
      gen->currentOutputY += window.outputH;
    }

    if (gen->currentOutputY >= gen->info->outputImageH) {
      gen->currentOutputX = -1;
      gen->currentOutputY = -1;
    }
  }
}

bool WindowGen_Valid(WindowGen *gen) {
  bool res = (gen->currentOutputX != -1 && gen->currentOutputY != -1);
  return res;
}

// ======================================
// Tensors

Tensor CreateTensor_NoAllocate(int64_t *dims, int numberDims) {
  Tensor tensor = {};
  tensor.dims.size = numberDims;

  int size = 1;
  for (int i = 0; i < numberDims; i++) {
    tensor.dims.data[i] = dims[i];
    size *= dims[i];
  }

  return tensor;
}

int Tensor_Size(Tensor tensor) {
  int size = 1;
  for (int i = 0; i < tensor.dims.size; i++) {
    size *= tensor.dims.data[i];
  }

  return size;
}

void Tensor_Print(Tensor tensor) {
  int size = Tensor_Size(tensor);

  for (int i = 0; i < tensor.dims.size; i++) {
    if (i != 0) {
      versat_printf("x ");
    }
    versat_printf("%d ", tensor.dims.data[i]);
  }
  versat_printf("\n");
  for (int i = 0; i < size; i++) {
    versat_printf("%f\n", tensor.data[i]);
  }
}

// ===============
//  Math utils

#define MATH_E 2.71828182846
#define MATH_PI 3.14159265359

double ABS(double x) { return x < 0 ? -x : x; }

typedef uint32_t u32;
typedef int32_t i32;

typedef uint8_t u8;

typedef struct {
  u32 mantissa : 23;
  u32 exponent : 8;
  u32 sign : 1;
} PackedFP;

typedef struct {
  u32 mantissa;
  i32 exponent;
  u8 sign;
} UnpackedFP;

UnpackedFP Unpack(float f, bool addImplicitBit) {
  PackedFP fp = *((PackedFP *)&f);

  UnpackedFP res = {};
  res.mantissa = fp.mantissa;

  if (addImplicitBit) {
    res.mantissa |= (1 << 23);
  }

  res.exponent = -127 + fp.exponent;
  res.sign = fp.sign;

  return res;
}

float Pack(UnpackedFP fp) {
  PackedFP res = {};
  res.sign = fp.sign;
  res.exponent = fp.exponent + 127;
  res.mantissa = fp.mantissa & ((1 << 24) - 1);

  return *((float *)&res);
}

static inline int Clamp(int min, int val, int max) {
  if (val < min) {
    return min;
  }
  if (val > max) {
    return max;
  }
  return val;
}

u32 Downsize(float f, int totalSize, int exponentAmount) {
  UnpackedFP unpacked = Unpack(f, false);

  int mantissaAmount = totalSize - 1 - exponentAmount;
  int signPos = totalSize - 1;

  u32 sign = (unpacked.sign) ? 1 : 0;

  int bias = (((1 << exponentAmount) / 2) - 1);

  u32 trueMantissa = Clamp(0, unpacked.mantissa >> (23 - mantissaAmount),
                           (1 << mantissaAmount) - 1);
  u32 trueExponent = Clamp(0, unpacked.exponent + (bias - 1), 31) &
                     ((1 << exponentAmount) - 1);

  u32 res =
      (trueMantissa | (trueExponent << mantissaAmount) | (sign << signPos));

  if (res >= (1 << totalSize)) {
    versat_printf("ERROR\n");
  }

  return res;
}

float Upsize(u32 in, int totalSize, int exponentAmount) {
  int mantissaAmount = totalSize - 1 - exponentAmount;

  u32 mantissaMask = ((1 << mantissaAmount) - 1);
  u32 exponentMask = ((1 << exponentAmount) - 1);

  u32 exponent = ((in >> mantissaAmount) & exponentMask);
  u32 mantissa = (in & mantissaMask) << (23 - mantissaAmount);
  int sign = ((in) >> (totalSize - 1)) & 1;

  bool subnormal = (exponent == 0);

  int bias = (((1 << exponentAmount) / 2) - 1);

  UnpackedFP unpacked = {};
  unpacked.sign = sign;
  unpacked.exponent = exponent - (bias - 1);
  unpacked.mantissa = mantissa;

  return Pack(unpacked);
}

void PrintBinary(u32 val) {
  for (int i = 31; i >= 0; i--) {
    u32 v = (val) & (1 << i);
    if (v) {
      versat_printf("1");
    } else {
      versat_printf("0");
    }
  }

  versat_printf("\n");
}

#define MAX_ITERS 20
double thetah_table[MAX_ITERS] = {
    0.549306, 0.255413, 0.125657, 0.062582, 0.031260, 0.015626, 0.007813,
    0.003906, 0.001953, 0.000977, 0.000488, 0.000244, 0.000122, 0.000061,
    0.000031, 0.000015, 0.000008, 0.000004, 0.000002, 0.000001};

double Cordic_arctanh(double y, double x) {
  double angle = 0.0;
  double P2i = 0.5;
  int k = 4;

  for (int i = 1; i < MAX_ITERS; i++) {
    double arc_tangent = thetah_table[i - 1];

    int iters = 1;
    if (i == k) {
      iters = 2;
      k = (3 * k) + 1;
    }

    for (int k = 0; k < iters; k++) {
      double sigma = 1.0;
      if (y < 0) {
        sigma = -1.0;
      }

      angle = angle + sigma * arc_tangent;
      double calcX = x - sigma * y * P2i;
      double calcY = y - sigma * x * P2i;

      x = calcX;
      y = calcY;
    }

    P2i /= 2.0;
  }

  return angle;
}

double Cordic_log(double in) {
  double val = in;

  if (in < 0.0f) {
    u32 minusInf = 0x7fffffff;
    return (double)VERSAT_CONVERT(minusInf, float);
  }

  double extra = 0;
  while (val > MATH_E) {
    val /= MATH_E;
    extra += 1;
  }

  while (val < 1.0f) {
    val *= MATH_E;
    extra -= 1;
  }

  double res = (2.0 * Cordic_arctanh(val - 1, val + 1)) + extra;
  return res;
}

double Cordic_exp(double exponent) {
  int k = 4;
  double angle = exponent;
  double y = 1.207497067763;
  double x = 1.207497067763;
  double P2i = 0.5;

  int extra = 0;
  while (angle > 1.0) {
    angle -= 1.0;
    extra += 1;
  }

  while (angle < 0) {
    angle += 1.0;
    extra -= 1;
  }

  for (int i = 1; i < MAX_ITERS; i++) {
    double arc_tangent = thetah_table[i - 1];

    int iters = 1;
    if (i == k) {
      iters = 2;
      k = (3 * k) + 1;
    }

    for (int k = 0; k < iters; k++) {
      double sigma = 1.0;
      if (angle > 0.0) {
        sigma = -1.0;
      }

      angle = angle + sigma * arc_tangent;
      double calcX = x - sigma * y * P2i;
      double calcY = y - sigma * x * P2i;

      x = calcX;
      y = calcY;
    }

    P2i /= 2.0;
  }

  for (int i = 0; i < extra; i++) {
    x *= MATH_E;
  }
  for (int i = 0; i > extra; i--) {
    x /= MATH_E;
  }

  return x;
}

double Cordic_pow(double b, double e) {
  double result = Cordic_exp(e * Cordic_log(b));
  return result;
}

u32 LogCalculateIndex(UnpackedFP unpacked) {
  // Hardcoded for a fractional precision of 12
  u32 fracIndex = (unpacked.mantissa >> 11);

  return fracIndex;
}

double Table_log(double in) {
  UnpackedFP unpacked = Unpack(in, false);

  float exp = (float)unpacked.exponent;
  u32 fracIndex = LogCalculateIndex(unpacked);

  float logMantissa = logMantissaTable[fracIndex];
  float result = exp * VERSAT_CONVERT(log2Val, float) + logMantissa;

  return result;
}

double Table_exp(double exponent) {
  float data = exponent;

  int expPrecision = 8;
  int halfExpPrecision = expPrecision - 1;
  int expMax = (1 << halfExpPrecision) - 1;

  // Hardware unit that calculates real part and also returns a floating point
  // "view" of the real part.
  int asInteger = (int)data;
  int realPart = asInteger;
  if (realPart >= expMax) {
    realPart = expMax;
  }
  if (realPart <= -expMax) {
    realPart = -expMax;
  }

  bool isNegative = false;
  if (data < 0.0) {
    realPart = -realPart;
    isNegative = true;
    data = -data;
  }

  // NOTE: If the realPart goes is outside the [-127,127] range, then
  //       we do not actually care about the fractional part since we are in the
  //       range of infinity anyway.

  // Should be between 0 and 1
  float asFraction = data - (float)realPart;
  u32 mantissaPartU32 = *((u32 *)&asFraction);

  UnpackedFP unpacked = Unpack(asFraction, true);

  // Hardware unit that performs this calculation and outputs the fracIndex.
  u32 fracIndex = (unpacked.mantissa >>
                   (8 + (16 - EXP_MANTISSA_PRECISION) - unpacked.exponent)) &
                  ((1 << EXP_MANTISSA_PRECISION) - 1);
  u32 expIndex = realPart + (isNegative ? 128 : 0);

  if (isNegative) {
    fracIndex += (1 << EXP_MANTISSA_PRECISION);
  }

  float exp = expTable[expIndex];
  float frac = expMantissaTable[fracIndex];

  float res = exp * frac;

  return res;
}

double Table_pow(double base, double power) {
  double result = Table_exp(power * Table_log(base));
  return result;
}

double Taylor_log(double in) {
  double start = 0.0;

  for (int i = 0; i < 1000; i++) {
    double last = start;
    start = start + 2.0 * ((in - Taylor_exp(start)) / (in + Taylor_exp(start)));
    if (last == start) {
      break;
    }
  }

  return start;
}

double Taylor_exp(double exponent) {
  double result = 1.0;
  double term = 1.0;
  double div = 1.0;
  int maxTerms = 100000; // Failsafe

  double lastResult = result;
  for (int n = 1; n < maxTerms; n++) {
    term *= exponent / div;
    div += 1.0;
    result += term;

    // As term gets smaller eventually we will reach a point where addition does
    // not change anything
    if (lastResult == result) {
      break;
    }
    lastResult = result;
  }

  return result;
}

double Taylor_pow(double base, double power) {
  double result = Taylor_exp(power * Taylor_log(base));
  return result;
}

#if !EMBED_TABLES
uint32_t *expMantissaTable;
uint32_t *expTable;
uint32_t *logMantissaTable;
#endif

#include <stdlib.h>

void Versat_Init() {
#if !EMBED_TABLES
  {
    logMantissaTable =
        (uint32_t *)malloc(sizeof(uint32_t) * LOG_MANTISSA_TABLE_SIZE);

    u32 increment =
        0x00000800; // TODO: Hardcoded for a fractionalPrecision of 12
    float f = 1.0f;
    for (int i = 0; i < LOG_MANTISSA_TABLE_SIZE; i++) {
      float val = Cordic_log(f);
      logMantissaTable[i] = VERSAT_CONVERT(val, u32);

      u32 asU32 = VERSAT_CONVERT(f, u32);
      asU32 += increment;
      f = VERSAT_CONVERT(asU32, float);
    }
  }

  {
    expTable = (uint32_t *)malloc(sizeof(uint32_t) * EXP_TABLE_SIZE);

    int halfSize = EXP_TABLE_SIZE / 2;

    float f = 0.0f;
    for (int i = 0; i < halfSize; i++) {
      float posExp = Cordic_exp(f);
      float negExp = Cordic_exp(-f);

      expTable[i] = VERSAT_CONVERT(posExp, u32);
      expTable[i + halfSize] = VERSAT_CONVERT(negExp, u32);
      f += 1.0f;
    }
  }

  {
    expMantissaTable =
        (uint32_t *)malloc(sizeof(uint32_t) * EXP_MANTISSA_TABLE_SIZE);

    float incr = 1.0 / (float)(1 << EXP_MANTISSA_PRECISION);
    float p = incr;

    float oneVal = 1.0f;

    expMantissaTable[0] = VERSAT_CONVERT(oneVal, u32);
    expMantissaTable[(1 << EXP_MANTISSA_PRECISION)] =
        VERSAT_CONVERT(oneVal, u32);

    for (int i = 1; i < (1 << EXP_MANTISSA_PRECISION); i++) {
      float posExp = Cordic_exp(p);
      float negExp = Cordic_exp(p);

      expMantissaTable[i] = VERSAT_CONVERT(posExp, u32);
      expMantissaTable[i + (1 << EXP_MANTISSA_PRECISION)] =
          VERSAT_CONVERT(negExp, u32);
      p += incr;
    }
  }
#endif
}

#if EMBED_TABLES
#define VERSAT_DO_EMBED_TABLES
#include "versat_embed_tables.h"

uint32_t *expMantissaTable = expMantissaTableArray;
uint32_t *expTable = expTableArray;
uint32_t *logMantissaTable = logMantissaTableArray;
#endif
