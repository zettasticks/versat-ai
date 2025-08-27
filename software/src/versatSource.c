#include "versat_ai.h"

#include "stdbool.h"
#include "stdint.h"
#include "stdlib.h" // REMOVE THIS AFTER REMOVING MALLOC AND FREE

#include "iob_printf.h"

#include "versat_accel.h"

#define MIN(A, B) (((A < B) ? (A) : (B)))
#define MAX(A, B) (((A > B) ? (A) : (B)))

void clear_cache();

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

  RunAccelerator(3);

  return output;
}

void *Versat_Relu(void *inputA, void *output, int index, ReluInfo *info) {
  printf("[Relu] Using Versat\n");

  volatile Top_ReluConfig *config = &accelConfig->Top_Relu;
  ActivateMergedAccelerator(MergeType_Top_Relu);

  int dims = info->dims;
  int64_t *dim = info->inputDims;

  int64_t totalSize = CalculateSizeOfDim(dim, dims);

  Linear_VRead(&config->input, inputA, totalSize);
  Linear_VWrite(&config->output, output, totalSize);

  RunAccelerator(3);

  return output;
}

void *Versat_Reshape(void *data, void *shape, void *output, int index,
                     ReshapeInfo *info) {
  printf("[Reshape] Using Versat\n");

  return data;
}

typedef struct {
  int x, y, c, width, height;
} Window;

static inline Window MaxPool_OutputToInput(Window outputSpace, int strideW,
                                           int strideH, int kernelW,
                                           int kernelH) {
  Window res = {};
  res.c = outputSpace.c;
  res.x = outputSpace.x * strideW;
  res.y = outputSpace.y * strideH;
  res.width = outputSpace.width * kernelW;
  res.height = outputSpace.height * kernelH;

  return res;
}

// Note: Only printing channel 0 and for input of the format NCHW
void PrintWindowValues(Window w, float *input, int imageW, int imageH) {
  printf("c-x,y:w,h = %d-%d,%d:%d,%d\n", w.c, w.x, w.y, w.width, w.height);

  int channelSize = imageW * imageH;
  for (int y = 0; y < w.height; y++) {
    for (int x = 0; x < w.width; x++) {
      int startY = y + w.y;
      int startX = x + w.x;

      int address = w.c * channelSize + startY * imageW + startX;

      printf("%7.4f ", input[address]);
    }
    printf("\n");
  }
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
      return;
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

    if (info->padding == PaddingType_SAME_LOWER) {
      res.leftPadW = res.padW;
      res.leftPadH = res.padH;
    } else {
      res.rightPadW = res.padW;
      res.rightPadH = res.padH;
    }
  }

  return res;
}

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

  int inputSizeC; // This is mostly the same as inputImageC since Conv cannot handle half sums. We must always process the entire input channels, we cannot handle half sums.

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

WindowGen StartWindowGen(ExtraInfo *info,bool iterateC,bool isNCHW) {
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

  printf("Output pos: %d:(%d,%d)\n", window.outputC, window.outputX, window.outputY);
  printf("Input pos: (%d,%d)\n", window.inputX, window.inputY);
  printf("WindowSize (Out view): %d %d %d\n",window.outputSizeC,window.outputH,window.outputW);
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

  // The only thing that we need to care about is the windows that are near padding regions
  // the fact that the accelerator must contain enough memory to support a window and
  // that we must make sure that the height of the window is stable. ( So
  // that we iterate over all the pixels correctly).
  res.outputW = 1;
  res.outputH = 1;

  // For now, just like the rest of the window, we only advance a single output channel
  res.outputSizeC = 1;

  // By default, input equals kernel size
  res.actualKernelW = gen->info->kernelW;
  res.actualKernelH = gen->info->kernelH;

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
  if (gen->currentOutputX == 0 && gen->info->leftPadW) {
    res.actualKernelW -= gen->info->leftPadW;
    res.kernelStartW = gen->info->leftPadW;
    res.padding |= PaddingRegion_LEFT;
  }
  if (gen->currentOutputX == gen->info->outputImageW - 1 &&
      gen->info->rightPadW) {
    res.actualKernelW -= gen->info->rightPadW;
    res.padding |= PaddingRegion_RIGHT;
  }

  if (gen->currentOutputY == 0 && gen->info->leftPadH) {
    res.actualKernelH -= gen->info->leftPadH;
    res.kernelStartH = gen->info->leftPadH;
    res.padding |= PaddingRegion_TOP;
  }
  if (gen->currentOutputY == gen->info->outputImageH - 1 &&
      gen->info->rightPadH) {
    res.actualKernelH -= gen->info->rightPadH;
    res.padding |= PaddingRegion_BOTTOM;
  }

  // Need to offset the input window if left padding exists.
  if (gen->currentOutputX > 0) {
    res.inputX -= gen->info->leftPadW;
  }
  if (gen->currentOutputY > 0) {
    res.inputY -= gen->info->leftPadH;
  }

  return res;
}

void WindowGen_Advance(WindowGen *gen) {
  AdvancedWindow window = WindowGen_Get(gen);

  if(gen->iterateC){
    if(gen->isNCHW){
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
      return;
    }

    res.leftPadW = info->padsDims[1];
    res.leftPadH = info->padsDims[0];

    res.rightPadW = info->padsDims[3];
    res.rightPadH = info->padsDims[2];

    res.padW = info->padsDims[1] + info->padsDims[3];
    res.padH = info->padsDims[0] + info->padsDims[2];
  } else if (info->padding == PaddingType_SAME_LOWER ||
             info->padding == PaddingType_SAME_UPPER) {
    res.padW = (res.outputImageW - 1) * res.strideW + res.kernelW - res.inputImageW;
    res.padH = (res.outputImageH - 1) * res.strideH + res.kernelH - res.inputImageH;

    if (info->padding == PaddingType_SAME_LOWER) {
      res.leftPadW = res.padW;
      res.leftPadH = res.padH;
    } else {
      res.rightPadW = res.padW;
      res.rightPadH = res.padH;
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

  MaxPool2D_VRead(&config->features, input, w.inputX, w.inputY, cInStart, w.actualKernelW,
                  w.actualKernelH, inputImageW, w.outputW, w.outputH, strideW,
                  strideH);
  Linear2_VWrite(&config->output, output, w.outputX, w.outputY, cOutStart, w.outputW,
                 w.outputH, outputImageW, stride);

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

  // MaxPool is currently using NCHW. We iterate by channels since there is no gain 
  // in passing a window that spans channels.

  // For MaxPool using NHWC the approach might be different. 
  // We might want to use windows that span channels 
  for (int c = 0; c < channels; c++) {
    WindowGen genInst = StartWindowGen(&extra,false,false);
    WindowGen *gen = &genInst;

    for (; WindowGen_Valid(gen); WindowGen_Advance(gen)) {
      AdvancedWindow w = WindowGen_Get(gen);
      //AdvancedWindow_Print(w);
      MaxPool_ProcessWindow(w, c, inputX, output, info);
    }
  }

  // Flush the remaining data from the accelerator
  RunAccelerator(2);

  return output;
}

Window Conv_OutputToInput(Window outputSpace, int strideW, int strideH,
                          int kernelW, int kernelH) {
  Window res = {};

  // TODO: Probably not working well
  res.c = outputSpace.c;
  res.x = outputSpace.x * strideW;
  res.y = outputSpace.y * strideH;
  res.width = outputSpace.width * kernelW;
  res.height = outputSpace.height * kernelH;

  return res;
}

void Conv_ProcessWindow(Window outSpace, void *inputX, void *inputW,
                        void *output, ConvInfo *info) {
  volatile Top_ConvConfig *config = &accelConfig->Top_Conv;

  // TODO(perf): All of these calculations could be pushed outside the loop.
  //             For now, we care more about correctness than performance.
  //             In fact, the vast majority of this code could be pushed to
  ExtraInfo extra = CalculateExtraInfo_Conv(info);

  int strideW = info->strideDims[1];
  int strideH = info->strideDims[0];

  int kernelW = info->kernelDims[1];
  int kernelH = info->kernelDims[0];

  int inputImageW = info->inputDims[3];
  int inputImageH = info->inputDims[2];
  int inputChannels = info->inputDims[1];

  int outputChannels = info->outputDims[1];
  int outputImageH = info->outputDims[2];
  int outputImageW = info->outputDims[3];

  int padW = 0;
  int padH = 0;
  int leftPadW = 0;
  int leftPadH = 0;

  if (info->padding == PaddingType_NOTSET) {
    // TODO: Need a better way of handling errors in this layer, I think.
    if (info->padsSize != 4) {
      printf("ERROR, pads size is not expected");
      return;
    }

    leftPadW = info->padsDims[1];
    leftPadH = info->padsDims[0];

    padW = info->padsDims[1] + info->padsDims[3];
    padH = info->padsDims[0] + info->padsDims[2];
  } else if (info->padding == PaddingType_SAME_LOWER ||
             info->padding == PaddingType_SAME_UPPER) {
    padW = (outputImageW - 1) * strideW + kernelW - inputImageW;
    padH = (outputImageH - 1) * strideH + kernelH - inputImageH;

    if (info->padding == PaddingType_SAME_LOWER) {
      leftPadW = padW;
      leftPadH = padH;
    }
  }

  Window inSpace =
      MaxPool_OutputToInput(outSpace, strideW, strideH, kernelW, kernelH);

  inSpace.x -= leftPadW;
  inSpace.y -= leftPadH;

  int weightStartX = 0;
  int weightStartY = 0;

  int sizeW = kernelW;
  int sizeH = kernelH;
  if (inSpace.x < 0) {
    int offset = -inSpace.x;
    inSpace.x += offset;
    weightStartX += offset;
    sizeW -= offset;
  } else if (inSpace.x + inSpace.width > inputImageW) {
    sizeW = MIN(kernelW, inputImageW - inSpace.x);
  }

  if (inSpace.y < 0) {
    int offset = -inSpace.y;
    inSpace.y += offset;
    weightStartY += offset;
    sizeH -= offset;
  } else if (inSpace.y + inSpace.height > inputImageH) {
    sizeH = MIN(kernelH, inputImageH - inSpace.y);
  }

  int stride = sizeW * sizeH * inputChannels;

  Conv2D_NHWC_VRead(&config->features, inputX, inSpace.x, inSpace.y, sizeW,
                    sizeH, inputImageW, inputChannels, outputChannels);
  Weight2D_VRead(&config->weights, inputW, weightStartX, weightStartY, sizeW,
                 sizeH, kernelW, kernelH, inputChannels, outputChannels);
  Linear2_NHWC_VWrite(&config->output, output, outSpace.x, outSpace.y,
                      outSpace.height, outSpace.width, 0, outputChannels,
                      outputImageW, stride);

  config->myAccum.strideMinusOne = stride - 1;
  EndAccelerator();
  StartAccelerator();
}

void Conv_ProcessWindow2(AdvancedWindow w, void *inputX, void *inputW,
                        void *output, ConvInfo *info) {
  volatile Top_ConvConfig *config = &accelConfig->Top_Conv;

  // TODO(perf): All of these calculations could be pushed outside the loop.
  //             For now, we care more about correctness than performance.
  //             In fact, the vast majority of this code could be pushed to
  ExtraInfo extra = CalculateExtraInfo_Conv(info);

  int inputImageW = info->inputDims[3];
  int inputImageH = info->inputDims[2];
  int inputImageC = info->inputDims[1];

  int outputImageW = info->outputDims[3];
  int outputImageH = info->outputDims[2];
  int outputImageC = info->outputDims[1];

  int kernelW = info->kernelDims[1];
  int kernelH = info->kernelDims[0];

  int stride = w.actualKernelW * w.actualKernelH * inputImageC;

  int strideW = info->strideDims[1];
  int strideH = info->strideDims[0];

  int sizeW = w.actualKernelW;
  int sizeH = w.actualKernelH;

  //Print_Conv2D_NHWC(w.inputX, w.inputY, sizeW,
  //                  sizeH, inputImageW, inputImageC, outputImageC);
  Conv2D_NHWC_VRead(&config->features, inputX, w.inputX, w.inputY, sizeW,
                    sizeH, inputImageW, inputImageC, outputImageC);
  Weight2D_VRead(&config->weights, inputW, w.kernelStartW, w.kernelStartH, sizeW,
                 sizeH, kernelW, kernelH, inputImageC, outputImageC);
  Linear2_NHWC_VWrite(&config->output, output, w.outputX, w.outputY,
                      w.outputH, w.outputW, 0, outputImageC,
                      outputImageW, stride);

  config->myAccum.strideMinusOne = stride - 1;
  EndAccelerator();
  StartAccelerator();
}

void *Versat_Conv(void *inputX, void *inputW, void *output, int index,
                  ConvInfo *info) {
  forceDoubleLoop = true;

  int inputChannels = info->inputDims[1];
  int inputImageW = info->inputDims[3];
  int inputImageH = info->inputDims[2];

  int outputChannels = info->outputDims[1];
  int outputImageH = info->outputDims[2];
  int outputImageW = info->outputDims[3];

  float *tempInput = (float *)malloc(sizeof(float) * inputImageW * inputImageH *
                                     inputChannels);
  float *tempOutput = (float *)malloc(sizeof(float) * outputImageW *
                                      outputImageH * outputChannels);

  float *inputView = (float *)inputX;
  // Convert NCHW to NHWC
  for (int y = 0; y < inputImageH; y++) {
    for (int x = 0; x < inputImageW; x++) {
      for (int c = 0; c < inputChannels; c++) {
        int NCHW_Index = c * (inputImageH * inputImageW) + y * inputImageW + x;
        int NHWC_Index = y * (inputImageW * inputChannels) + x * inputChannels + c;

        tempInput[NHWC_Index] = inputView[NCHW_Index];
      }
    }
  }

  ActivateMergedAccelerator(MergeType_Top_Conv);

#if 1
  ExtraInfo extra = CalculateExtraInfo_Conv(info);

  WindowGen genInst = StartWindowGen(&extra,true,true);
  WindowGen *gen = &genInst;

  for (; WindowGen_Valid(gen); WindowGen_Advance(gen)) {
    AdvancedWindow w = WindowGen_Get(gen);
    //AdvancedWindow_Print(w);
    
    Conv_ProcessWindow2(w,tempInput,inputW,tempOutput,info);
    //MaxPool_ProcessWindow(w, c, inputX, output, info);
  }
#endif

#if 0
  for (int c = 0; c < outputChannels; c++) {
    for (int y = 0; y < outputImageH; y++) {
      for (int x = 0; x < outputImageW; x++) {
        Window outSpace = {};

        outSpace.x = x;
        outSpace.y = y;
        outSpace.c = c;
        outSpace.width = 1;
        outSpace.height = 1;

        Conv_ProcessWindow(outSpace, tempInput, inputW, tempOutput, info);
      }
    }
  }
#endif

  // Flush the remaining data from the accelerator
  RunAccelerator(2);

  float *outputView = (float *)output;
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

  return output;
}
