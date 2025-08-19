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
  //             In fact, the vast majority of this code could be pushed to the python code generation
  //             function
  int strideW = info->strideDims[1];
  int strideH = info->strideDims[0];

  int kernelW = info->kernelDims[1];
  int kernelH = info->kernelDims[0];

  int inputImageW = info->inputDims[3];
  int inputImageH = info->inputDims[2];
  int channels = info->inputDims[1];

  int padW = 0;
  int padH = 0;
  int outputImageH = 0;
  int outputImageW = 0;
  int leftPadW = 0;
  int leftPadH = 0;

  if(info->padding == PaddingType_NOTSET){
    // TODO: Need a better way of handling errors in this layer, I think.
    if(info->padsSize != 4){
      printf("ERROR, pads size is not expected");
      return;
    }

    leftPadW = info->padsDims[0];
    leftPadH = info->padsDims[1];

    padW = info->padsDims[0] + info->padsDims[2];
    padH = info->padsDims[1] + info->padsDims[3];

    outputImageW = (inputImageW + padW - kernelW) / strideW + 1;
    outputImageH = (inputImageH + padH - kernelH) / strideH + 1;

    outputImageH = MAX(outputImageH,1);
    outputImageW = MAX(outputImageW,1);
  } else if(info->padding == PaddingType_SAME_LOWER || info->padding == PaddingType_SAME_UPPER){
    outputImageH = (inputImageH + strideH - 1) / strideH;
    outputImageW = (inputImageW + strideW - 1) / strideW;

    outputImageH = MAX(outputImageH,1);
    outputImageW = MAX(outputImageW,1);

    padW = (outputImageW - 1) * strideW + kernelW - inputImageW;
    padH = (outputImageH - 1) * strideH + kernelH - inputImageH;

    if(info->padding == PaddingType_SAME_LOWER){
      leftPadW = padW;
      leftPadH = padH;
    }
  } else if(info->padding == PaddingType_VALID){
    outputImageH = (inputImageH - kernelH + 1 + (strideH - 1)) / strideH;
    outputImageW = (inputImageW - kernelW + 1 + (strideW - 1)) / strideW;

    outputImageH = MAX(outputImageH,1);
    outputImageW = MAX(outputImageW,1);
  }

  int totalSizePerChannelInput = inputImageH * inputImageW;
  int totalSizePerChannelOutput = outputImageH * outputImageW;

  Window inSpace = MaxPool_OutputToInput(outSpace,strideW,strideH,kernelW,kernelH);

  inSpace.x -= leftPadW;
  inSpace.y -= leftPadH;

  int actualKernelW = kernelW;
  int actualKernelH = kernelH;
  if(inSpace.x < 0){
    int offset = -inSpace.x;
    inSpace.x += offset;
    actualKernelW -= offset;
  } else if(inSpace.x + inSpace.width > inputImageW){
    actualKernelW = MIN(kernelW,inputImageW - inSpace.x);
  }

  if(inSpace.y < 0){
    int offset = -inSpace.y;
    inSpace.y += offset;
    actualKernelH -= offset;
  } else if(inSpace.y + inSpace.height > inputImageH){
    actualKernelH = MIN(kernelH,inputImageH - inSpace.y);
  }

  int inSizeW = inSpace.width;
  int inSizeH = inSpace.height;
  int outSizeW = outSpace.width;
  int outSizeH = outSpace.height;
  int cInStart = outSpace.c * totalSizePerChannelInput;
  int cOutStart = outSpace.c * totalSizePerChannelOutput;
  int stride = actualKernelH * actualKernelW;

  MaxPool2D_VRead(&config->features, inputX, inSpace.x, inSpace.y, cInStart, actualKernelW,
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

  int padW = 0;
  int padH = 0;
  int outputImageH = 0;
  int outputImageW = 0;

  // TODO: This code is basically just a repeat of the above code.
  //       With some things removed that are not needed.
  //       Vast majority of this code is duplicated when in reality it could just be calculated once in here and then passed as an argument to the proper function.
  //       In reality, a lot of this code is also "bad" because we could just push this into the python generator code and avoid doing these calculations at runtime.
  //       Remember, embedded is much slower and we do not want to spend time doing these operations.
  if(info->padding == PaddingType_NOTSET){
    // TODO: Need a better way of handling errors in this layer, I think.
    if(info->padsSize != 4){
      printf("ERROR, pads size is not expected");
      return output;
    }

    padW = info->padsDims[0] + info->padsDims[2];
    padH = info->padsDims[1] + info->padsDims[3];

    outputImageW = (inputImageW + padW - kernelW) / strideW + 1;
    outputImageH = (inputImageH + padH - kernelH) / strideH + 1;

    outputImageH = MAX(outputImageH,1);
    outputImageW = MAX(outputImageW,1);
  } else if(info->padding == PaddingType_SAME_LOWER || info->padding == PaddingType_SAME_UPPER){
    outputImageH = (inputImageH + strideH - 1) / strideH;
    outputImageW = (inputImageW + strideW - 1) / strideW;

    outputImageH = MAX(outputImageH,1);
    outputImageW = MAX(outputImageW,1);

    padW = (outputImageW - 1) * strideW + kernelW - inputImageW;
    padH = (outputImageH - 1) * strideH + kernelH - inputImageH;
  } else if(info->padding == PaddingType_VALID){
    outputImageH = (inputImageH - kernelH + 1 + (strideH - 1)) / strideH;
    outputImageW = (inputImageW - kernelW + 1 + (strideW - 1)) / strideW;

    outputImageH = MAX(outputImageH,1);
    outputImageW = MAX(outputImageW,1);
  }

  //printf("OutputSize:%d %d\n",outputImageH,outputImageW);

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

Window Conv_OutputToInput(Window outputSpace,int strideW,int strideH,int kernelW,int kernelH){
  Window res = {};

  // TODO: Probably not working well 
  res.c = outputSpace.c;
  res.x = outputSpace.x * strideW;
  res.y = outputSpace.y * strideH;
  res.width = outputSpace.width * kernelW;
  res.height = outputSpace.height * kernelH;

  return res;
}

void Conv_ProcessWindow(Window outSpace,void* inputX,void* inputW,void* output,ConvInfo* info){
  volatile Top_ConvConfig *config = &accelConfig->Top_Conv;

  // TODO(perf): All of these calculations could be pushed outside the loop.
  //             For now, we care more about correctness than performance.
  //             In fact, the vast majority of this code could be pushed to
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

  if(info->padding == PaddingType_NOTSET){
    // TODO: Need a better way of handling errors in this layer, I think.
    if(info->padsSize != 4){
      printf("ERROR, pads size is not expected");
      return;
    }

    leftPadW = info->padsDims[0];
    leftPadH = info->padsDims[1];

    padW = info->padsDims[0] + info->padsDims[2];
    padH = info->padsDims[1] + info->padsDims[3];
  } else if(info->padding == PaddingType_SAME_LOWER || info->padding == PaddingType_SAME_UPPER){
    padW = (outputImageW - 1) * strideW + kernelW - inputImageW;
    padH = (outputImageH - 1) * strideH + kernelH - inputImageH;

    printf("Padding:%d-%d\n",padW,padH);

    if(info->padding == PaddingType_SAME_LOWER){
      leftPadW = padW;
      leftPadH = padH;
    }
  }

  Window inSpace = MaxPool_OutputToInput(outSpace,strideW,strideH,kernelW,kernelH);

  inSpace.x -= leftPadW;
  inSpace.y -= leftPadH;

  int weightStartX = 0;
  int weightStartY = 0;

  int sizeW = kernelW;
  int sizeH = kernelH;
  if(inSpace.x < 0){
    int offset = -inSpace.x;
    inSpace.x += offset;
    weightStartX += offset;
    sizeW -= offset;
  } else if(inSpace.x + inSpace.width > inputImageW){
    sizeW = MIN(kernelW,inputImageW - inSpace.x);
  }

  if(inSpace.y < 0){
    int offset = -inSpace.y;
    inSpace.y += offset;
    weightStartY += offset;
    sizeH -= offset;
  } else if(inSpace.y + inSpace.height > inputImageH){
    sizeH = MIN(kernelH,inputImageH - inSpace.y);
  }

  int stride = sizeW * sizeH * inputChannels;

  Conv2D_NHWC_VRead(&config->features,inputX,inSpace.x,inSpace.y,sizeW,sizeH,inputImageW,inputChannels,outputChannels);
  Weight2D_VRead(&config->weights,inputW,weightStartX,weightStartY,sizeW,sizeH,kernelW,kernelH,inputChannels,outputChannels);
  Linear2_NHWC_VWrite(&config->output,output,outSpace.x,outSpace.y,outSpace.height,outSpace.width,0,outputChannels,outputImageW,stride);

  config->myAccum.strideMinusOne = stride - 1;
  EndAccelerator();
  StartAccelerator();
}

void *Versat_Conv(void *inputX, void *inputW, void *output, int index,
                    ConvInfo *info){
  forceDoubleLoop = true;

  int inputChannels = info->inputDims[1];
  int inputImageW = info->inputDims[3];
  int inputImageH = info->inputDims[2];

  int outputChannels = info->outputDims[1];
  int outputImageH = info->outputDims[2];
  int outputImageW = info->outputDims[3];

  float* tempInput = (float*) malloc(sizeof(float) * inputImageW * inputImageH * inputChannels);
  float* tempOutput =  (float*) malloc(sizeof(float) * outputImageW * outputImageH * outputChannels);

  float* inputView = (float*) inputX;
  // Convert NCHW to NHWC
  for(int y = 0; y < inputImageH; y++){
    for(int x = 0; x < inputImageW; x++){
      for(int c = 0; c < inputChannels; c++){
        int NCHW_Index = c * (inputImageH * inputImageW)   + y * inputImageW   + x;
        int NHWC_Index = y * (inputImageW * inputChannels) + x * inputChannels + c;

        tempInput[NHWC_Index] = inputView[NCHW_Index];
      }
    }
  }

#if 0
  printf("Features\n");
  for(int i = 0; i < 4; i++){
    printf("%5.2f\n",inputView[i]);
  }
  printf("\n");
  printf("Weights\n");
  float* weights = (float*) inputW;
  for(int i = 0; i < 9; i++){
    printf("%5.2f\n",weights[i]);
  }
#endif

  ActivateMergedAccelerator(MergeType_Top_Conv);

  for(int c = 0; c < outputChannels; c++){
    for(int y = 0; y < outputImageH; y++){
      for(int x = 0; x < outputImageW; x++){
        Window outSpace = {};

        outSpace.x = x;
        outSpace.y = y;
        outSpace.c = c;
        outSpace.width = 1;
        outSpace.height = 1;

        Conv_ProcessWindow(outSpace,tempInput,inputW,tempOutput,info);
      }    
    }
  }

  // Flush the remaining data from the accelerator
  RunAccelerator(2);

  float* outputView = (float*) output;
  // Convert NHWC to NCHW 
  for(int c = 0; c < outputChannels; c++){
    for(int y = 0; y < outputImageH; y++){
      for(int x = 0; x < outputImageW; x++){
        int NCHW_Index = c * (outputImageH * outputImageW)   + y * outputImageW   + x;
        int NHWC_Index = y * (outputImageW * outputChannels) + x * outputChannels + c;

        outputView[NCHW_Index] = tempOutput[NHWC_Index];
      }
    }
  }

  free(tempInput);
  free(tempOutput);

  printf("%p\n",output);

  return output;

  //return Software_Conv(inputX,inputW,output,index,info);

  // TODO: We probably want to cleanup a bit the maxpool logic, maybe simplify some parts.
  //       Before doing the same for convolutions.
  //       The biggest thing to simplify is the way we go from the output window to the input window
  //       while taking into account padding and all the different types of stride, kernel size, dilations
  //       and so on.
  //       Maxpool is not handling this stuff very cleanly, I think we can do better.

}
