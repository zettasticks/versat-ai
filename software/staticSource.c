#include "versat_ai.h"

#include "stdint.h"

// TODO: Eventually remove this, cannot depend on OS specific code.
#include "stdio.h"

typedef char bool;

size_t GetOutputMemorySize(){return 100;}
size_t GetTemporaryMemorySize(){return 100;}

typedef struct{
   // Attributes as described by onnx.
   int auto_pad;
   int* dilations;
   int group;
   int* kernel_shape;
   int* pads;
   int* strides;

   // Extra info to help 
   int** inputDims;
} ConvInfo;

typedef struct{
   // Extra info to help 
   int** inputDims;
   bool needToBroadcast;
} AddInfo;

// TODO: In fact, because we have all the information that we need, we do not even have to 
//       keep the size of the arrays of the model. Everything is already know.
//       It eventually boils down to how much of this we want to generate and how much of this we want to calculate at runtime.

int GetAmountOfInitializers(void* base){
   int* view = (int*) base;
   return *view;
}

int* GetInitializersSize(void* base){
   int* view = (int*) base;
   return view + 1;
}

// TODO: This is actually not needed. We can precompute this.
void* GetInitializer(void* base,int index){
   int amount = GetAmountOfInitializers(base);
   int* sizes = GetInitializersSize(base);

   int* view = (int*) base;
   // Skips the initializers sizes values and array
   view += 1;
   view += amount;

   for(int i = 0; i < index; i++){
      view += sizes[i];
   }

   return view;
}

#define OFFSET_PTR(PTR,OFFSET) ((void*) (((char*) PTR) + OFFSET))

void* Conv(void* inputX,void* inputW,void* output){
  return NULL;
}

void* Reshape(void* data,void* shape,void* output){
  int64_t* view = (int64_t*) shape;

  printf("%ld %ld\n",view[0],view[1]);
  
  return NULL;
}

void* Add(void* inputA,void* inputB,void* output){
  return NULL;
}

void* Relu(void* inputX,void* output){
  return NULL;
}

void* MaxPool(void* inputX,void* output){
  return NULL;
}

void* MatMul(void* inputA,void* inputB,void* output){
  return NULL;
}

void AssertAlmostEqual(void* toTest,void* correctValues){
  
}

InferenceOutput RunInference(void* outputMemory,void* temporaryMemory,void** input,void* modelMemory){
  return (InferenceOutput){};
}

InferenceOutput DebugRunInference(void* outputMemory,void* temporaryMemory,void** inputs,void* modelMemory,void* correctInput){
  void* res_0 = Reshape(OFFSET_PTR(modelMemory,0),OFFSET_PTR(modelMemory,10240),OFFSET_PTR(temporaryMemory,25088));
  AssertAlmostEqual(res_0,OFFSET_PTR(correctInput,0));
  void* res_1 = Conv(inputs[0],OFFSET_PTR(modelMemory,10256),OFFSET_PTR(temporaryMemory,35328));
  AssertAlmostEqual(res_1,OFFSET_PTR(correctInput,10240));
  void* res_2 = Add(OFFSET_PTR(correctInput,10240),OFFSET_PTR(modelMemory,11056),OFFSET_PTR(temporaryMemory,0));
  AssertAlmostEqual(res_2,OFFSET_PTR(correctInput,35328));
  void* res_3 = Relu(OFFSET_PTR(correctInput,35328),OFFSET_PTR(temporaryMemory,35328));
  AssertAlmostEqual(res_3,OFFSET_PTR(correctInput,60416));
  void* res_4 = MaxPool(OFFSET_PTR(correctInput,60416),OFFSET_PTR(temporaryMemory,0));
  AssertAlmostEqual(res_4,OFFSET_PTR(correctInput,85504));
  void* res_5 = Conv(OFFSET_PTR(correctInput,85504),OFFSET_PTR(modelMemory,11088),OFFSET_PTR(temporaryMemory,35328));
  AssertAlmostEqual(res_5,OFFSET_PTR(correctInput,91776));
  void* res_6 = Add(OFFSET_PTR(correctInput,91776),OFFSET_PTR(modelMemory,23888),OFFSET_PTR(temporaryMemory,47872));
  AssertAlmostEqual(res_6,OFFSET_PTR(correctInput,104320));
  void* res_7 = Relu(OFFSET_PTR(correctInput,104320),OFFSET_PTR(temporaryMemory,0));
  AssertAlmostEqual(res_7,OFFSET_PTR(correctInput,116864));
  void* res_8 = MaxPool(OFFSET_PTR(correctInput,116864),OFFSET_PTR(temporaryMemory,24064));
  AssertAlmostEqual(res_8,OFFSET_PTR(correctInput,129408));
  void* res_9 = Reshape(OFFSET_PTR(correctInput,129408),OFFSET_PTR(modelMemory,23952),OFFSET_PTR(temporaryMemory,23040));
  AssertAlmostEqual(res_9,OFFSET_PTR(correctInput,130432));
  void* res_10 = MatMul(OFFSET_PTR(correctInput,130432),OFFSET_PTR(correctInput,0),OFFSET_PTR(temporaryMemory,0));
  AssertAlmostEqual(res_10,OFFSET_PTR(correctInput,131456));
  void* res_11 = Add(OFFSET_PTR(correctInput,131456),OFFSET_PTR(modelMemory,23968),OFFSET_PTR(outputMemory,0));
  AssertAlmostEqual(res_11,OFFSET_PTR(correctInput,131496));

  return (InferenceOutput){};
}
