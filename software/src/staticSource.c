#include "versat_ai.h"

#include "stdbool.h"
#include "stdint.h"

#include "iob_printf.h"

#define exit(...) ((void)0)
void clear_cache();

#define OFFSET_PTR(PTR, OFFSET) ((void *)(((char *)PTR) + OFFSET))

#define MIN(A, B) ((A < B ? A : B))
#define MAX(A, B) ((A > B ? A : B))

typedef struct {
  int addressVars[16];
  int64_t *iterationDims;
  int64_t *properDims;
  int numberDims;
} AddressGen;

void Print(AddressGen *gen) {
  for (int i = 0; i < gen->numberDims; i++) {
    if (i != 0) {
      printf(" x ");
    }
    printf("%d", gen->addressVars[i]);
  }

  printf(" [");
  for (int i = 0; i < gen->numberDims; i++) {
    if (i != 0) {
      printf(" x ");
    }
    printf("%ld", gen->iterationDims[i]);
  }
  printf("]\n");
}

int GetAddress(AddressGen *gen) {
  int address = 0;
  for (int i = 0; i < gen->numberDims; i++) {
    int index = gen->addressVars[i];

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

bool IsValid(AddressGen *gen) {
  // Is this the only thing that we need?
  if (gen->addressVars[0] >= gen->iterationDims[0]) {
    return false;
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

void Advance(AddressGen *gen) {
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

AddressGen StartAddress(int64_t *iterationDims, int64_t *properDims,
                        int numberDims) {
  AddressGen gen = {};
  gen.iterationDims = iterationDims;
  gen.properDims = properDims;
  gen.numberDims = numberDims;

  return gen;
}

void *Software_Conv(void *inputX, void *inputW, void *output, int index,
                    ConvInfo *info) {
  int batchSize = info->inputDims[0];
  int inChannels = info->inputDims[1];
  int inH = info->inputDims[2];
  int inW = info->inputDims[3];

  int featureMaps = info->kernelDims[0];
  int outputChannelsPerGroup = info->kernelDims[1];
  int kernelH = info->kernelDims[2];
  int kernelW = info->kernelDims[3];

  int outChannels = info->outDims[1]; // Should be equal to feature maps.
  int outH = info->outDims[2];
  int outW = info->outDims[3];

  float *input = (float *)inputX;
  float *kernel = (float *)inputW;
  float *outView = (float *)output;

  bool print = false;

  for (int outC = 0; outC < outChannels; outC++) {
    for (int y = 0; y < outH; y++) {
      for (int x = 0; x < outW; x++) {
        float accum = 0.0f;
        for (int inC = 0; inC < inChannels; inC++) {
          if (print)
            printf("Going to accumulate: %d %d\n", y, x);

          for (int ky = 0; ky < kernelH; ky++) {
            for (int kx = 0; kx < kernelW; kx++) {
              int deltaX = x + kx - (kernelH / 2);
              int deltaY = y + ky - (kernelH / 2);
              int inIndex = inC * info->inputDims[3] * info->inputDims[2] +
                            deltaY * info->inputDims[3] + deltaX;
              float inputVal;

              if (deltaX < 0 || deltaY < 0 || deltaX >= inW || deltaY >= inH) {
                inputVal = 0.0f;
              } else {
                inputVal = input[inIndex];
              }

              int inK = outC * info->kernelDims[3] * info->kernelDims[2] *
                        info->kernelDims[1];
              inK += inC * info->kernelDims[3] * info->kernelDims[2];
              inK += ky * info->kernelDims[3];
              inK += kx;

              if (print)
                printf("%d %f %d\n", inIndex, inputVal, inK);

              float kernelVal = kernel[inK];
              accum += inputVal * kernelVal;
            }
          }
        }
        int outPos = outC * info->outDims[3] * info->outDims[2];
        outPos += y * info->outDims[3];
        outPos += x;

        outView[outPos] = accum;

        if (print)
          exit(0);
      }
    }
  }

  return output;
}

void *Software_Reshape(void *data, void *shape, void *output, int index,
                       ReshapeInfo *info) {
  int64_t *shapeDims = (int64_t *)shape;

  AddressGen in =
      StartAddress(info->inputDims, info->inputDims, info->numberInputDims);
  AddressGen out = StartAddress(shapeDims, shapeDims, info->numberShapeDims);

  float *inView = (float *)data;
  float *outView = (float *)output;

  // TODO: Looking at the indexes produced, it appears that the Reshape
  // operation does not need to copy data around, altought I still need to look
  // further into this.
  while (IsValid(&out)) {
    int inAddr = GetAddress(&in);
    int outAddr = GetAddress(&out);

    Advance(&in);
    Advance(&out);

    // printf("%d %d\n",outAddr,inAddr);

    outView[outAddr] = inView[inAddr];
  }

  return output;
}

void *Software_Add(void *inputA, void *inputB, void *output, int index,
                   AddInfo *info) {
  float *viewA = (float *)inputA;
  float *viewB = (float *)inputB;
  float *out = (float *)output;

  AddressGen inA =
      StartAddress(info->broadCastedShape, info->firstInputDim, info->maxDims);
  AddressGen inB =
      StartAddress(info->broadCastedShape, info->secondInputDim, info->maxDims);
  AddressGen outGen = StartAddress(info->broadCastedShape,
                                   info->broadCastedShape, info->maxDims);

  while (IsValid(&outGen)) {
    int indexA = GetAddress(&inA);
    int indexB = GetAddress(&inB);
    int indexO = GetAddress(&outGen);

    // printf("%d %d %d\n",indexA,indexB,indexO);

    Advance(&inA);
    Advance(&inB);
    Advance(&outGen);

    float valA = viewA[indexA];
    float valB = viewB[indexB];

    out[indexO] = valA + valB;
    // printf("%f   %f   %f\n",viewA[indexA],viewB[indexB],out[indexO]);
  }

  return output;
}

void *Software_Relu(void *inputX, void *output, int index, ReluInfo *info) {
  float *view = (float *)inputX;
  float *out = (float *)output;

  int64_t totalSize = CalculateSizeOfDim(info->inputDims, info->dims);

  for (int64_t i = 0; i < totalSize; i++) {
    float val = view[i];
    out[i] = MAX(0.0f, val);
  }

  return output;
}

void *Software_MaxPool(void *inputX, void *output, int index,
                       MaxPoolInfo *info) {
  // Currently software impl of maxpool is hardcoded to run the downloadable
  // example.
  float *view = (float *)inputX;
  float *out = (float *)output;

  // Code for 4 tensors, we probably can make it more generic without a lot of changes.
  if (info->dims == 4) {
    for (int a = 0; a < info->outputDims[0]; a++) {
      for (int b = 0; b < info->outputDims[1]; b++) {
        for (int c = 0; c < info->outputDims[2]; c++) {
          for (int d = 0; d < info->outputDims[3]; d++) {
            int ia =
                info->inputDims[3] * info->inputDims[2] * info->inputDims[1];
            int ib = info->inputDims[3] * info->inputDims[2];
            int ic = info->inputDims[3];

            int oa =
                info->outputDims[3] * info->outputDims[2] * info->outputDims[1];
            int ob = info->outputDims[3] * info->outputDims[2];
            int oc = info->outputDims[3];

            float max = 0.0f;
            bool firstSet = false;
            for(int y = 0; y < info->kernelDims[0]; y++){
              for(int x = 0; x < info->kernelDims[1]; x++){
                int trueY = (c * info->strideDims[0] + y);
                int trueX = (d * info->strideDims[1] + x);

                // Padding, we just skip ahead since we use a flag to indicate validity
                if(trueY >= info->inputDims[2]){
                  continue;
                } else if(trueX >= info->inputDims[3]){
                  continue;
                }

                int index = a * ia + b * ib + trueY * ic + trueX;

                float val = view[index];

                // NOTE: Padding should never affect the output value.
                //       Because negative values exist, we cannot just use zero to represent a padded value.
                //       We might get away with using -inf, but for now we just use an extra flag to check validity.
                if(!firstSet){
                  max = val;
                  firstSet = true;
                } else {
                  max = MAX(max,val);
                }
              }
            }

            int outputIndex = a * oa + b * ob + c * oc + d;
            out[outputIndex] = max;
          }
        }
      }
    }
  }

  return output;
}

void *Software_MatMul(void *inputA, void *inputB, void *output, int index,
                      MatMulInfo *info) {
  float *viewA = (float *)inputA;
  float *viewB = (float *)inputB;
  float *viewOut = (float *)output;

  for (int y = 0; y < info->outputDims[0]; y++) {
    for (int x = 0; x < info->outputDims[1]; x++) {
      int indexOut = info->outputDims[0] * y + x;

      viewOut[indexOut] = 0.0f;
      for (int c = 0; c < 256; c++) {
        int indexA = info->inputADims[1] * y + c;
        int indexB = info->inputBDims[1] * c + x;

        float valA = viewA[indexA];
        float valB = viewB[indexB];

        viewOut[indexOut] += valA * valB;
      }
    }
  }

  return output;
}

static inline float absf(float a) {
  if (a < 0.0f) {
    return -a;
  }
  return a;
}

int64_t CalculateSizeOfDim(int64_t *dim, int dims) {
  int64_t size = 1;
  for (int i = 0; i < dims; i++) {
    size *= dim[i];
  }

  return size;
}

void AssertAlmostEqual(void *toTest, void *correctValues, int index) {
  float *test = (float *)toTest;
  float *correct = (float *)correctValues;

  size_t outputSize = layers[index].outputSize / sizeof(float);

  if (outputSize == 0) {
    printf("Error, AssertAlmostEqual with output size of 0. Should not be "
           "possible\n");
    return;
  }

  int maxIncorrect = 10;
  bool printOk = true;

  // Make sure that cache is not affecting the verification process
  clear_cache();

  int incorrectFound = 0;
  for (int i = 0; i < outputSize; i++) {
    if (absf(correct[i] - test[i]) > 0.001f) {
      if (incorrectFound == 0) {
        printf("\n");
        printf("[%s] (Layer %d) FAIL:\n", layers[index].typeName, index);
      }
      printf("  Index: %4d Different values %.4f %.4f\n", i, correct[i],
             test[i]);
      incorrectFound += 1;
    }

    if (incorrectFound >= maxIncorrect) {
      printf("More than %d incorrect found, quitting early\n", maxIncorrect);
      printf("\n");
      break;
    }
  }

  if (printOk && incorrectFound == 0) {
    printf("[%s] (Layer %d) - OK\n", layers[index].typeName, index);
  }
}

InferenceOutput RunInference(void *outputMemory, void *temporaryMemory,
                             void **input, void *modelMemory) {
  return (InferenceOutput){};
}
