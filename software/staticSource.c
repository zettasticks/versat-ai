#include "versat_ai.h"

#include "stdint.h"

// TODO: Eventually remove this, cannot depend on OS specific code.
#include "stdio.h"

typedef char bool;


#define OFFSET_PTR(PTR,OFFSET) ((void*) (((char*) PTR) + OFFSET))

void* Conv(void* inputX,void* inputW,void* output,int index,ConvInfo* info){
  return output;
}

void* Reshape(void* data,void* shape,void* output,int index,ReshapeInfo* info){
  //int64_t* view = (int64_t*) shape;

  //printf("%ld %ld\n",view[0],view[1]);
  
  return output;
}

#define MIN(A,B) ((A < B ? A : B))
#define MAX(A,B) ((A > B ? A : B))

void* Add(void* inputA,void* inputB,void* output,int index,AddInfo* info){
  float* viewA = (float*) inputA;
  float* viewB = (float*) inputB;
  float* out = (float*) output;

   // Code for 4 tensors
  if(info->maxDims == 4){
      for(int a = 0; a < info->broadCastedShape[0]; a++){
         for(int b = 0; b < info->broadCastedShape[1]; b++){
            for(int c = 0; c < info->broadCastedShape[2]; c++){
               for(int d = 0; d < info->broadCastedShape[3]; d++){
                  int da = info->firstInputDim[3] * info->firstInputDim[2] * info->firstInputDim[1];
                  int db = info->firstInputDim[3] * info->firstInputDim[2];
                  int dc = info->firstInputDim[3];

                  int xa = info->secondInputDim[3] * info->secondInputDim[2] * info->secondInputDim[1];
                  int xb = info->secondInputDim[3] * info->secondInputDim[2];
                  int xc = info->secondInputDim[3];

                  int oa = info->broadCastedShape[3] * info->broadCastedShape[2] * info->broadCastedShape[1];
                  int ob = info->broadCastedShape[3] * info->broadCastedShape[2];
                  int oc = info->broadCastedShape[3];

                  int ma = MIN(info->secondInputDim[0],a);
                  int mb = MIN(info->secondInputDim[1],b);
                  int mc = MIN(info->secondInputDim[2],c);
                  int md = MIN(info->secondInputDim[3],d);

                  int indexA =  a * da +  b * db +  c * dc +  d;
                  int indexB = ma * xa + mb * xb + mc * xc + md;
                  int indexO =  a * oa +  b * ob +  c * oc +  d;

                  float valA = viewA[indexA];
                  float valB = viewB[indexB];

                  out[indexO] = valA + valB;
               }
            }   
         }   
      }
   } else if(info->maxDims == 2){
      for(int a = 0; a < info->broadCastedShape[0]; a++){
         for(int b = 0; b < info->broadCastedShape[1]; b++){
            int da = info->firstInputDim[1];
            int xa = info->secondInputDim[1];
            int oa = info->broadCastedShape[1];

            int ma = MIN(info->secondInputDim[0],a);
            int mb = MIN(info->secondInputDim[1],b);

            int indexA =  a * da +  b;
            int indexB = ma * xa + mb;
            int indexO =  a * oa +  b;

            float valA = viewA[indexA];
            float valB = viewB[indexB];

            out[indexO] = valA + valB;
         }   
      }      
   }

  return output;
}

void* Relu(void* inputX,void* output,int index,ReluInfo* info){
   float* view = (float*) inputX;
   float* out = (float*) output;

   int totalSize = 1;
   for(int i = 0; i < info->dims; i++){
      totalSize *= info->inputDims[i];
   }

   for(int i = 0; i < totalSize; i++){
      float val = view[i];
      out[i] = MAX(0.0f,val);      
   }

  return output;
}

void* MaxPool(void* inputX,void* output,int index,MaxPoolInfo* info){
  return output;
}

void* MatMul(void* inputA,void* inputB,void* output,int index,MatMulInfo* info){
  return output;
}

static inline float absf(float a){
   if(a < 0.0f){
      return -a;
   }
   return a;
}

void AssertAlmostEqual(void* toTest,void* correctValues,int index){
   float* test = (float*) toTest;
   float* correct = (float*) correctValues;

   size_t outputSize = layers[index].outputSize / sizeof(float);

   int incorrectFound = 0;
   for(int i = 0; i < outputSize; i++){
      if(absf(correct[i] - test[i]) > 1.0f){
         printf("Layer(%d)[%s] Index: %d\n",index,layers[index].typeName,i);
         printf("  Different values %f %f\n",correct[i],test[i]);
         incorrectFound += 1;
      }

      if(incorrectFound > 10){
         printf("More than 10 incorrect found, quitting early\n");
         break;
      }
   }

   if(incorrectFound == 0){
      printf("Fully checked and found no errors for Layer(%d)[%s]\n",index,layers[index].typeName);
   }
}

InferenceOutput RunInference(void* outputMemory,void* temporaryMemory,void** input,void* modelMemory){
  return (InferenceOutput){};
}
