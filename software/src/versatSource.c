#include "versat_ai.h"

#include "stdbool.h"
#include "stdint.h"
#include "stdlib.h" // REMOVE THIS AFTER REMOVING MALLOC AND FREE

#include "iob_printf.h"

#include "versat_accel.h"

#define MIN(A, B) (((A < B) ? (A) : (B)))
#define MAX(A, B) (((A > B) ? (A) : (B)))

void clear_cache();

typedef union {
  iptr i;
  float f;
} Convertor;

iptr NoConvert(float f) {
  Convertor c = {};
  c.f = f;
  return c.i;
}

static inline int64_t GetDim(int64_t *dimArray, int dimSize, int index) {
  if (index < dimSize) {
    return MAX(dimArray[index], 1);
  }

  return 1;
}

static inline int64_t GetSize(int64_t *dimArray, int dimSize, int index) {
  if (index < dimSize) {
    if (dimArray[index] > 1) {
      return 1;
    }
  }

  return 0;
}

void *Versat_Add(void *inputA, void *inputB, void *output, int index,
                 AddInfo *info) {
  printf("[Add] Using Versat\n");

  int64_t *l = info->firstInputDim;
  int64_t *r = info->secondInputDim;
  int64_t *o = info->broadCastedShape;
  int d = info->maxDims;

  volatile Top_AddConfig *config = &accelConfig->Top_Add;

  ActivateMergedAccelerator(MergeType_Top_Add);

  DataBroadCasted_VRead(
      &config->inputs_0, inputA, GetSize(l, d, 0), GetDim(o, d, 0),
      GetSize(l, d, 1), GetDim(o, d, 1), GetDim(l, d, 1), GetSize(l, d, 2),
      GetDim(o, d, 2), GetDim(l, d, 2), GetSize(l, d, 3), GetDim(o, d, 3),
      GetDim(l, d, 3), GetSize(l, d, 4), GetDim(o, d, 4), GetDim(l, d, 4),
      GetSize(l, d, 5), GetDim(o, d, 5), GetDim(l, d, 5));

  DataBroadCasted_VRead(
      &config->inputs_1, inputB, GetSize(r, d, 0), GetDim(o, d, 0),
      GetSize(r, d, 1), GetDim(o, d, 1), GetDim(r, d, 1), GetSize(r, d, 2),
      GetDim(o, d, 2), GetDim(r, d, 2), GetSize(r, d, 3), GetDim(o, d, 3),
      GetDim(r, d, 3), GetSize(r, d, 4), GetDim(o, d, 4), GetDim(r, d, 4),
      GetSize(r, d, 5), GetDim(o, d, 5), GetDim(r, d, 5));

  DataSimple_VWrite(&config->output, output, GetDim(o, d, 0), GetDim(o, d, 1),
                    GetDim(o, d, 2), GetDim(o, d, 3), GetDim(o, d, 4),
                    GetDim(o, d, 5));

  RunAccelerator(1);
  config->inputs_0.enabled = 0;
  config->inputs_1.enabled = 0;
  config->output.enabled = 0;
  RunAccelerator(2);

  return output;
}

void *Versat_Relu(void *inputA, void *output, int index, ReluInfo *info) {
  printf("[Relu] Using Versat\n");

  volatile Top_ReluConfig *config = &accelConfig->Top_Relu;
  ActivateMergedAccelerator(MergeType_Top_Relu);

  int dims = info->dims;
  int64_t *dim = info->inputDims;

  int64_t totalSize = CalculateSizeOfDim(dim, dims);

  int64_t maxAtATime = 256;

  float *inputView = (float *)inputA;
  float *outputView = (float *)output;

  for (int i = 0; i < totalSize; i += maxAtATime) {
    int size = MIN(maxAtATime, totalSize - i);

    Linear_VRead(&config->input, &inputView[i], size);
    Linear_VWrite(&config->output, &outputView[i], size);

    RunAccelerator(1);
  }

  config->input.enabled = 0;
  config->output.enabled = 0;
  RunAccelerator(2);

  return output;
}

void *Versat_Reshape(void *data, void *shape, void *output, int index,
                     ReshapeInfo *info) {
  printf("[Reshape] Using Versat\n");

  return data;
}

typedef struct {
  int strideW;
  int strideH;

  int kernelW;
  int kernelH;

  int inputImageW;
  int inputImageH;
  int inputImageC;

  int outputImageW;
  int outputImageH;
  int outputImageC;

  int leftPadW;
  int leftPadH;
  int rightPadW;
  int rightPadH;

  int padW;
  int padH;
} ExtraInfo;

typedef struct {
  ExtraInfo *info;
  int currentOutputC;
  int currentOutputX;
  int currentOutputY;
  bool iterateC;
  bool isNCHW; // Otherwise assume it is NHWC
} WindowGen;

typedef enum {
  PaddingRegion_TOP = (1 << 1),
  PaddingRegion_LEFT = (1 << 2),
  PaddingRegion_RIGHT = (1 << 3),
  PaddingRegion_BOTTOM = (1 << 4)
} PaddingRegion;

typedef struct {
  int inputX;
  int inputY;
  int outputX;
  int outputY;

  int startC;
  int inputSizeC; // This is mostly the same as inputImageC since Conv cannot
                  // handle half sums. We must always process the entire input
                  // channels, we cannot handle half sums.

  int outputC;
  int outputSizeC;

  int actualKernelW;
  int actualKernelH;

  int kernelStartW;
  int kernelStartH;

  int outputW;
  int outputH;

  PaddingRegion padding;
} AdvancedWindow;

WindowGen StartWindowGen(ExtraInfo *info, bool iterateC, bool isNCHW) {
  WindowGen res = {};
  res.info = info;
  res.iterateC = iterateC;
  res.isNCHW = isNCHW;
  return res;
}

void AdvancedWindow_Print(AdvancedWindow window) {
  bool printedOnce = false;
  if (window.padding & PaddingRegion_TOP) {
    printf("Pad_TOP");
    printedOnce = true;
  }
  if (window.padding & PaddingRegion_BOTTOM) {
    if (printedOnce) {
      printf(" | ");
    }
    printf("Pad_BOTTOM");
    printedOnce = true;
  }
  if (window.padding & PaddingRegion_LEFT) {
    if (printedOnce) {
      printf(" | ");
    }
    printf("Pad_LEFT");
    printedOnce = true;
  }
  if (window.padding & PaddingRegion_RIGHT) {
    if (printedOnce) {
      printf(" | ");
    }
    printf("Pad_RIGHT");
    printedOnce = true;
  }

  printf("\n");

  printf("Output pos: %d:(%d,%d)\n", window.outputC, window.outputX,
         window.outputY);
  printf("Input pos: (%d,%d)\n", window.inputX, window.inputY);
  printf("WindowSize (Out view): %d %d %d\n", window.outputSizeC,
         window.outputH, window.outputW);
  printf("KernelSizeAndOffset: %d:%d - %d:%d\n", window.actualKernelW,
         window.kernelStartW, window.actualKernelH, window.kernelStartH);
}

AdvancedWindow WindowGen_Get(WindowGen *gen) {
  AdvancedWindow res = {};

  res.outputX = gen->currentOutputX;
  res.outputY = gen->currentOutputY;
  res.outputC = gen->currentOutputC;

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
  res.outputSizeC = 1;

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
    // printf("A: %d %d %d
    // %d\n",res.inputX,res.actualKernelW,gen->info->inputImageW,offset);
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
    // printf("B: %d %d %d
    // %d\n",res.inputY,res.actualKernelH,gen->info->inputImageH,offset);
    res.actualKernelH -= offset;
    res.padding |= PaddingRegion_BOTTOM;
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
      printf("ERROR, pads size is not expected");
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
    res.padW =
        (res.outputImageW - 1) * res.strideW + res.kernelW - res.inputImageW;
    res.padH =
        (res.outputImageH - 1) * res.strideH + res.kernelH - res.inputImageH;

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

static inline void MaxPool_ProcessWindow(AdvancedWindow w, int channel,
                                         void *input, void *output,
                                         MaxPoolInfo *info) {
  volatile Top_MaxpoolConfig *config = &accelConfig->Top_Maxpool;

  int inputImageW = info->inputDims[3];
  int inputImageH = info->inputDims[2];

  int outputImageW = info->outputDims[3];
  int outputImageH = info->outputDims[2];

  int cInStart = channel * inputImageH * inputImageW;
  int cOutStart = channel * outputImageH * outputImageW;

  int stride = w.actualKernelW * w.actualKernelH;

  int strideW = info->strideDims[1];
  int strideH = info->strideDims[0];

  MaxPool2D_VRead(&config->features, input, w.inputX, w.inputY, cInStart,
                  w.actualKernelW, w.actualKernelH, inputImageW, w.outputW,
                  w.outputH, strideW, strideH);
  Linear2_VWrite(&config->output, output, w.outputX, w.outputY, cOutStart,
                 w.outputW, w.outputH, outputImageW, stride);

  config->accum.strideMinusOne = stride - 1;
  EndAccelerator();
  StartAccelerator();
}

// Currently hardcoded for 2D kernels.
void *Versat_MaxPool(void *inputX, void *output, int index, MaxPoolInfo *info) {
  forceDoubleLoop = true;
  volatile Top_MaxpoolConfig *config = &accelConfig->Top_Maxpool;
  ActivateMergedAccelerator(MergeType_Top_Maxpool);

  int channels = info->inputDims[1];

  ExtraInfo extra = CalculateExtraInfo_MaxPool(info);

  // MaxPool is currently using NCHW. We iterate by channels since there is no
  // gain in passing a window that spans channels.

  // For MaxPool using NHWC the approach might be different.
  // We might want to use windows that span channels
  for (int c = 0; c < channels; c++) {
    WindowGen genInst = StartWindowGen(&extra, false, false);
    WindowGen *gen = &genInst;

    for (; WindowGen_Valid(gen); WindowGen_Advance(gen)) {
      AdvancedWindow w = WindowGen_Get(gen);
      MaxPool_ProcessWindow(w, c, inputX, output, info);
    }
  }

  // Flush the remaining data from the accelerator
  config->features.enabled = 0;
  config->output.enabled = 0;
  RunAccelerator(2);

  return output;
}

#if 1
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
      printf("ERROR, pads size is not expected");
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
    res.padW =
        (res.outputImageW - 1) * res.strideW + res.kernelW - res.inputImageW;
    res.padH =
        (res.outputImageH - 1) * res.strideH + res.kernelH - res.inputImageH;

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

static inline void AveragePool_ProcessWindow(AdvancedWindow w, int channel,
                                             void *input, void *output,
                                             AveragePoolInfo *info) {
  volatile Top_AveragePoolConfig *config = &accelConfig->Top_AveragePool;

  int inputImageW = info->inputDims[3];
  int inputImageH = info->inputDims[2];

  int outputImageW = info->outputDims[3];
  int outputImageH = info->outputDims[2];

  int cInStart = channel * inputImageH * inputImageW;
  int cOutStart = channel * outputImageH * outputImageW;

  int stride = w.actualKernelW * w.actualKernelH;

  int strideW = info->strideDims[1];
  int strideH = info->strideDims[0];

  MaxPool2D_VRead(&config->features, input, w.inputX, w.inputY, cInStart,
                  w.actualKernelW, w.actualKernelH, inputImageW, w.outputW,
                  w.outputH, strideW, strideH);
  Linear2_VWrite(&config->output, output, w.outputX, w.outputY, cOutStart,
                 w.outputW, w.outputH, outputImageW, stride);

  config->averagePool_accum.strideMinusOne = stride - 1;
  config->invertedDivisor.constant = NoConvert(1.0f / (float)stride);
  EndAccelerator();
  StartAccelerator();
}

// Currently hardcoded for 2D kernels.
void *Versat_AveragePool(void *inputX, void *output, int index,
                         AveragePoolInfo *info) {
  forceDoubleLoop = true;
  volatile Top_AveragePoolConfig *config = &accelConfig->Top_AveragePool;
  ActivateMergedAccelerator(MergeType_Top_AveragePool);

  int channels = info->inputDims[1];

  ExtraInfo extra = CalculateExtraInfo_AveragePool(info);

  // Using NHWC
  for (int c = 0; c < channels; c++) {
    WindowGen genInst = StartWindowGen(&extra, false, false);
    WindowGen *gen = &genInst;

    for (; WindowGen_Valid(gen); WindowGen_Advance(gen)) {
      AdvancedWindow w = WindowGen_Get(gen);
      AveragePool_ProcessWindow(w, c, inputX, output, info);
    }
  }

  // Flush the remaining data from the accelerator
  config->features.enabled = 0;
  config->output.enabled = 0;
  RunAccelerator(2);

  return output;
}
#endif

typedef struct {
  Dimensions dims;
  float* data;
} Tensor;

Tensor CreateTensor(int64_t* dims,int numberDims){
  Tensor tensor = {};
  tensor.dims.size = numberDims;

  int size = 1;
  for(int i = 0; i < numberDims; i++){
    tensor.dims.data[i] = dims[i];
    size *= dims[i];
  }

  tensor.data = (float*) malloc(sizeof(float) * size);
  return tensor;
}

int Tensor_Size(Tensor tensor){
  int size = 1;
  for(int i = 0; i < tensor.dims.size; i++){
    size *= tensor.dims.data[i];
  }

  return size;
}

void Tensor_Print(Tensor tensor){
  int size = Tensor_Size(tensor);

  for(int i = 0; i < tensor.dims.size; i++){
    if(i != 0){
      printf("x ");
    }
    printf("%d ",tensor.dims.data[i]);
  }
  printf("\n");
  for(int i = 0; i < size; i++){
    printf("%f\n",tensor.data[i]);
  }
}

Tensor Tensor_ExtractView(Tensor input,int dimIndex,int start,int size){
  AddressGen in = StartAddress(input.dims.data,input.dims.data,input.dims.size);

  in.offsetAddressVars[dimIndex] = start;
  in.iterationDims[dimIndex] = (int64_t) size;

  Dimensions outDims = input.dims;
  outDims.data[dimIndex] = size;

  Tensor output = CreateTensor(outDims.data,outDims.size);

  AddressGen out = StartAddress(outDims.data,outDims.data,outDims.size);

  for(; Address_IsValid(&in); Address_Advance(&in),Address_Advance(&out)){
    int inIndex = Address_GetValue(&in);
    int outIndex = Address_GetValue(&out);

    output.data[outIndex] = input.data[inIndex];
  }

  return output;
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
      printf("ERROR, pads size is not expected");
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
    res.padW =
        (res.outputImageW - 1) * res.strideW + res.kernelW - res.inputImageW;
    res.padH =
        (res.outputImageH - 1) * res.strideH + res.kernelH - res.inputImageH;

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

void ConvWithBias_ProcessWindow(AdvancedWindow w, void *inputX, void *inputW,
                                void *output, float *bias, ConvInfo *info,int inputC,int outputC) {
  volatile Top_ConvConfig *config = &accelConfig->Top_Conv;

  int inputImageW = info->inputDims[3];
  int inputImageC = inputC;

  int outputImageW = info->outputDims[3];
  int outputImageC = outputC;

  printf("Input imageC:%d\n",inputImageC);

  int kernelW = info->kernelDims[1];
  int kernelH = info->kernelDims[0];

  int stride = w.actualKernelW * w.actualKernelH * inputImageC;

  int convChannelSize = inputImageC;
  int kernelChannelSize = inputImageC;
  int group = info->group;
  int convStartC = w.startC;

  // MARK: Effects of group are being simulated right now. Do not put final logic 
  //convChannelSize /= group;

  Conv2D_NHWC_VRead(&config->features, inputX, w.inputX, w.inputY,
                    w.actualKernelW, w.actualKernelH, inputImageW, inputImageC,convChannelSize,
                    outputImageC, convStartC);
  Weight2D_VRead(&config->weights, inputW, w.kernelStartW, w.kernelStartH,
                 w.actualKernelW, w.actualKernelH, kernelW, kernelH,
                 inputImageC,kernelChannelSize, outputImageC, convStartC);
  Linear2_NHWC_VWrite(&config->output, output, w.outputX, w.outputY, w.outputH,
                      w.outputW, 0, outputImageC, outputImageW, stride);

  if(bias == NULL){
    static float bias = 0.0f;
    LinearStrided_VRead(&config->bias, &bias, 1, 1);
  } else {
    LinearStrided_VRead(&config->bias, bias, outputImageC, stride);
  }

  config->myAccum.strideMinusOne = stride - 1;

  EndAccelerator();
  StartAccelerator();
}

void *Versat_Conv(void *inputX, void *inputW, void *output, int index,
                  ConvInfo *info) {
  Versat_ConvWithBias(inputX,inputW,NULL,output,index,info);
}

void *Versat_ConvWithBias(void *inputX, void *inputW, void *inputB,
                          void *output, int index, ConvInfo *info) {
  forceDoubleLoop = true;

  printf("[Versat Conv]\n");

  volatile Top_ConvConfig *config = &accelConfig->Top_Conv;

  int batches = info->inputDims[0];
  int inputChannels = info->inputDims[1];
  int inputImageW = info->inputDims[3];
  int inputImageH = info->inputDims[2];

  int outputChannels = info->outputDims[1];
  int outputImageH = info->outputDims[2];
  int outputImageW = info->outputDims[3];

  int inputSize = inputImageW * inputImageH * inputChannels;
  int outputSize = outputImageW * outputImageH * outputChannels;
  int group = info->group;

  ActivateMergedAccelerator(MergeType_Top_Conv);

  ExtraInfo extra = CalculateExtraInfo_Conv(info);

  for(int i = 0; i < batches; i++){
    Tensor tempInputTensor = CreateTensor(info->inputDims,4);
    Tensor tempOutputTensor = CreateTensor(info->outputDims,4);

    int64_t kernelDims[] = {2,1,2,2};
    Tensor kernel = CreateTensor(kernelDims,4);
    kernel.data = inputW;

    printf("Kernel:\n");
    Tensor_Print(kernel);

    float* tempInput = tempInputTensor.data;
    float* tempOutput = tempOutputTensor.data;

    float *inputView = (float *)inputX;
    float *biasView = (float *)inputB;

    inputView += i * inputSize;

    // Convert NCHW to NHWC
    for (int y = 0; y < inputImageH; y++) {
      for (int x = 0; x < inputImageW; x++) {
        for (int c = 0; c < inputChannels; c++) {
          int NCHW_Index = c * (inputImageH * inputImageW) + y * inputImageW + x;
          int NHWC_Index =
              y * (inputImageW * inputChannels) + x * inputChannels + c;

          tempInput[NHWC_Index] = inputView[NCHW_Index];
        }
      }
    }

    Tensor_Print(tempInputTensor);

    // Extract the channel
    Tensor extracted1 = Tensor_ExtractView(tempInputTensor,3,0,1);
    Tensor extracted2 = Tensor_ExtractView(tempInputTensor,3,1,1);

    printf("Extracted1:\n");
    Tensor_Print(extracted1);
    printf("Extracted2:\n");
    Tensor_Print(extracted2);

    if(group == 2){
      // MARK: Simulation the effects of group
      extra.inputImageC = 1;
      extra.outputImageC = 1;

      Dimensions dims = CreateDimensions(info->inputDims,4);
      dims.data[1] = 1;

      int size = Dimensions_Size(dims); 

      Tensor tempOutputTensor2 = CreateTensor(dims.data,4);
      Tensor tempOutputTensor3 = CreateTensor(dims.data,4);

      float* tempOutput1 = tempOutputTensor2.data; 
      float* tempOutput2 = tempOutputTensor3.data; 

      WindowGen genInst = StartWindowGen(&extra, true, true);
      WindowGen *gen = &genInst;

      for (; WindowGen_Valid(gen); WindowGen_Advance(gen)) {
        AdvancedWindow w = WindowGen_Get(gen);
        AdvancedWindow_Print(w);
        ConvWithBias_ProcessWindow(w, extracted1.data, ((float*) inputW), tempOutput1, biasView,info,1,1);
      }

      genInst = StartWindowGen(&extra, true, true);

      for (; WindowGen_Valid(gen); WindowGen_Advance(gen)) {
        AdvancedWindow w = WindowGen_Get(gen);
        ConvWithBias_ProcessWindow(w, extracted2.data, ((float*) inputW) + size, tempOutput2, biasView,info,1,1);
      }

      // Flush the remaining data from the accelerator
      config->features.enabled = 0;
      config->weights.enabled = 0;
      config->output.enabled = 0;
      config->bias.enabled = 0;
      RunAccelerator(2);

      int index = 0;
      for(int i = 0; i < outputSize / 2; i++){
        tempOutput[index++] = tempOutputTensor2.data[i];
      }
      for(int i = 0; i < outputSize / 2; i++){
        tempOutput[index++] = tempOutputTensor3.data[i];
      }

      printf("First1:\n");
      Tensor_Print(tempOutputTensor2);
      printf("First2:\n");
      Tensor_Print(tempOutputTensor3);
    } else {

      WindowGen genInst = StartWindowGen(&extra, true, true);
      WindowGen *gen = &genInst;

      for (; WindowGen_Valid(gen); WindowGen_Advance(gen)) {
        AdvancedWindow w = WindowGen_Get(gen);
        ConvWithBias_ProcessWindow(w, tempInput, inputW, tempOutput, biasView,info,info->inputDims[1],info->outputDims[1]);
      }

      // Flush the remaining data from the accelerator
      config->features.enabled = 0;
      config->weights.enabled = 0;
      config->output.enabled = 0;
      config->bias.enabled = 0;
      RunAccelerator(2);
    }

    float *outputView = (float *)output;
    outputView += i * outputSize;

    // Convert NHWC to NCHW
    for (int c = 0; c < outputChannels; c++) {
      for (int y = 0; y < outputImageH; y++) {
        for (int x = 0; x < outputImageW; x++) {
          int NCHW_Index =
              c * (outputImageH * outputImageW) + y * outputImageW + x;
          int NHWC_Index =
              y * (outputImageW * outputChannels) + x * outputChannels + c;

          outputView[NCHW_Index] = tempOutput[NHWC_Index];
        }
      }
    }

    free(tempInput);
    free(tempOutput);
  }

  return output;
}

void *Versat_MatMul(void *inputA, void *inputB, void *output, int index,
                    MatMulInfo *info) {

  ActivateMergedAccelerator(MergeType_Top_MatMul);
  volatile Top_MatMulConfig *config = &accelConfig->Top_MatMul;

  int totalBSize = info->inputBDims[0] * info->inputBDims[1];

  float *tempB = malloc(sizeof(float) * totalBSize);

  float *viewA = (float *)inputA;
  float *viewB = (float *)inputB;
  float *viewOut = (float *)output;

  int AH = info->inputADims[0];
  int AW = info->inputADims[1];

  int BH = info->inputBDims[0];
  int BW = info->inputBDims[1];

  int OH = info->outputDims[0];
  int OW = info->outputDims[1];

  if (AW != BH) {
    printf("Something very wrong is happening in MatMul\n");
  }

  // After transpose, matrix goes from size BH,BW to BW,BH
  for (int y = 0; y < BH; y++) {
    for (int x = 0; x < BW; x++) {
      // Transposing B
      tempB[x * BH + y] = viewB[y * BW + x];
    }
  }

  for (int y = 0; y < OH; y++) {
    for (int x = 0; x < OW; x++) {
      float *lineAStart = &viewA[y * AW];
      float *lineBStart = &tempB[x * AW];

      float *out = &viewOut[y * OW + x];

      Linear_VRead(&config->leftRow, lineAStart, AW);
      Linear_VRead(&config->rightRow, lineBStart, BH);
      LinearStrided_VWrite(&config->output, out, 1, AW);

      config->myAccum.strideMinusOne = AW - 1;

      EndAccelerator();
      StartAccelerator();
    }
  }

  config->leftRow.enabled = 0;
  config->rightRow.enabled = 0;
  config->output.enabled = 0;
  RunAccelerator(2);

  free(tempB);

  return output;
}

void *Versat_Softmax(void *inputA, void *output, int index, SoftmaxInfo *info) {
  return Software_Softmax(inputA, output, index, info);
}
