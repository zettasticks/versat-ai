#include "versat_ai.h"

#include "stdbool.h"
#include "stdint.h"

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

typedef struct{
  int x,y,c,width,height;
} Window;

static inline Window MaxPool_OutputToInput(Window outputSpace,int strideW,int strideH,int kernelW,int kernelH){
  Window res = {};
  res.c = outputSpace.c;
  res.x = outputSpace.x * strideW;
  res.y = outputSpace.y * strideH;
  res.width = outputSpace.width * kernelW;
  res.height = outputSpace.height * kernelH;

  return res;
}

void PrintWindowValues(Window w,float* input,int imageW,int imageH){
  printf("c-x,y:w,h = %d-%d,%d:%d,%d\n",w.c,w.x,w.y,w.width,w.height);

  int channelSize = imageW * imageH;
  for(int y = 0; y < w.height; y++){
    for(int x = 0; x < w.width; x++){
      int startY = y + w.y;
      int startX = x + w.x;

      int address = w.c * channelSize + startY * imageW + startX;

      printf("%7.4f ",input[address]);
    }
    printf("\n");
  }
}

static inline void MaxPool_ProcessWindow(Window outSpace,void* inputX,void* output,MaxPoolInfo *info){
  volatile Top_MaxpoolConfig *config = &accelConfig->Top_Maxpool;

  // TODO(perf): All of these calculations could be pushed outside the loop.
  //             For now, we care more about correctness than performance.
  int strideW = info->strideDims[1];
  int strideH = info->strideDims[0];

  int kernelW = info->kernelDims[1];
  int kernelH = info->kernelDims[0];

  int inputImageW = info->inputDims[3];
  int inputImageH = info->inputDims[2];
  int channels = info->inputDims[1];

  int outputImageH = inputImageH / strideH;
  int outputImageW = inputImageW / strideW;

  // Need to handle this special case for the notset where kernel is larger than the image size.
  if(info->padding == PaddingType_NOTSET){
    if(kernelW > inputImageW){
      outputImageW = 1;
    }
    if(kernelH > inputImageH){
      outputImageH = 1;
    }

    kernelW = MIN(kernelW,inputImageW);
    kernelH = MIN(kernelH,inputImageH);

    strideW = MIN(strideW,inputImageW);
    strideH = MIN(strideH,inputImageH);
  } else {
    outputImageH = (inputImageH + strideH - 1) / strideH;
    outputImageW = (inputImageW + strideW - 1) / strideW;
  }

  int totalSizePerChannelInput = inputImageH * inputImageW;
  int totalSizePerChannelOutput = outputImageH * outputImageW;

  Window inSpace = MaxPool_OutputToInput(outSpace,strideW,strideH,kernelW,kernelH);

  int actualKernelW = MIN(kernelW,inputImageW - inSpace.x);
  int actualKernelH = MIN(kernelH,inputImageH - inSpace.y);

  int inSizeW = inSpace.width;
  int inSizeH = inSpace.height;
  int outSizeW = outSpace.width;
  int outSizeH = outSpace.height;
  int cInStart = outSpace.c * totalSizePerChannelInput;
  int cOutStart = outSpace.c * totalSizePerChannelOutput;
  int stride = actualKernelH * actualKernelW;

  // NOTE: The reason that we use outSpace.x and .y is because the address gen already calculates the correct input position from an outputSpace POV.
  MaxPool2D_VRead(&config->features, inputX, outSpace.x, outSpace.y, cInStart, actualKernelW,
                  actualKernelH, inputImageW, inSizeW, inSizeH, strideW, strideH);
  Linear2_VWrite(&config->output, output, outSpace.x,outSpace.y, cOutStart, outSizeW, outSizeH, outputImageW, stride);

  config->accum.strideMinusOne = stride - 1;
  EndAccelerator();
  StartAccelerator();
}

// Currently hardcoded for 2D kernels.
void *Versat_MaxPool(void *inputX, void *output, int index, MaxPoolInfo *info) {
  forceDoubleLoop = true;
  volatile Top_MaxpoolConfig *config = &accelConfig->Top_Maxpool;
  ActivateMergedAccelerator(MergeType_Top_Maxpool);

#if 1
  int strideW = info->strideDims[1];
  int strideH = info->strideDims[0];

  int kernelW = info->kernelDims[1];
  int kernelH = info->kernelDims[0];

  int inputImageW = info->inputDims[3];
  int inputImageH = info->inputDims[2];
  int channels = info->inputDims[1];

  int outputImageH = inputImageH / strideH;
  int outputImageW = inputImageW / strideW;

  if(info->padding == PaddingType_NOTSET){
    if(kernelW > inputImageW){
      outputImageW = 1;
    }
    if(kernelH > inputImageH){
      outputImageH = 1;
    }

    kernelW = MIN(kernelW,inputImageW);
    kernelH = MIN(kernelH,inputImageH);

    strideW = MIN(strideW,inputImageW);
    strideH = MIN(strideH,inputImageH);
  } else {
    outputImageH = (inputImageH + strideH - 1) / strideH;
    outputImageW = (inputImageW + strideW - 1) / strideW;
  }

  printf("OutputSize:%d %d\n",outputImageH,outputImageW);

  for(int c = 0; c < channels; c++){
    for(int y = 0; y < outputImageH; y++){
      for(int x = 0; x < outputImageW; x++){
        Window outSpace = {};

        outSpace.x = x;
        outSpace.y = y;
        outSpace.c = c;
        outSpace.width = 1;
        outSpace.height = 1;

        MaxPool_ProcessWindow(outSpace,inputX,output,info);
      }    
    }
  }
#else
  int inputImageW = info->inputDims[3];
  int inputImageH = info->inputDims[2];

  Window all = {};
  all.width = inputImageW;
  all.height = inputImageH;

  printf("Entire image\n");  
  PrintWindowValues(all,inputX,inputImageW,inputImageH);

  Window outSpace = {};

  outSpace.x = 0;
  outSpace.y = 0;
  outSpace.c = 0;
  outSpace.width = 1;
  outSpace.height = 1;

  MaxPool_ProcessWindow(outSpace,inputX,output,info);
#endif

  // Flush the remaining data from the accelerator
  RunAccelerator(2);

  return output;
}