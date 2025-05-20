#ifndef INCLUDED_VERSAT_AI
#define INCLUDED_VERSAT_AI

#include <stddef.h>

typedef char bool;

// Memory is allocated by the user. 

// These functions are constant, meaning that buffers can be reused if needed 
// TODO: Maybe replace this by an extern variable? Function just to return a value is overkill.
size_t GetOutputMemorySize();
size_t GetTemporaryMemorySize();
size_t GetModelMemorySize();
size_t GetCorrectMemorySize();

#define OFFSET_PTR(PTR,OFFSET) ((void*) (((char*) PTR) + OFFSET))

// TODO: Remove this.
// What we should have is a function that goes:
// OutputInfo GetOutput(void* output,int index);
// That not only fixes the output to the correct address but also fills it with some information.

typedef struct{
   int numberOfOutputs;
   void** outputLocation; // outputLocation[N] gives the address of the N output.
} InferenceOutput;

typedef struct{
   const char* name;
   const char* typeName;
   size_t outputSize;
} LayerInfo;

typedef struct{
   // Extra info to help 
   int maxDims;
   int* firstInputDim;
   int* secondInputDim;
   int* broadCastedShape;
} AddInfo;

typedef struct{
   int dims;
   int* inputDims;
} ReluInfo;

typedef struct{
   // Attributes as described by onnx.
   int auto_pad;
   int* dilations;
   int group;
   int* kernel_shape;
   int* pads;
   int* strides;

   // Extra info to help 
   int inputs;
   int* inputDims;
} ConvInfo;

typedef struct{} ReshapeInfo;
typedef struct{} MaxPoolInfo;
typedef struct{} MatMulInfo;

extern LayerInfo layers[];
extern int numberLayers;

// Operations
void* Conv(void* inputX,void* inputW,void* output,int index,ConvInfo* info);
void* Reshape(void* data,void* shape,void* output,int index,ReshapeInfo* info);
void* Add(void* inputA,void* inputB,void* output,int index,AddInfo* info);
void* Relu(void* inputX,void* output,int index,ReluInfo* info);
void* MaxPool(void* inputX,void* output,int index,MaxPoolInfo* info);
void* MatMul(void* inputA,void* inputB,void* output,int index,MatMulInfo* info);
void AssertAlmostEqual(void* toTest,void* correctValues,int index);

// Output and Temporary memory must be allocated by the user. Call the GetXXXMemorySize functions to obtain the amount of memory required.
// Model memory is memory that contains a loaded model (provided by a binary file that must be read at runtime).
InferenceOutput RunInference(void* outputMemory,void* temporaryMemory,void** inputs,void* modelMemory);
InferenceOutput DebugRunInference(void* outputMemory,void* temporaryMemory,void** inputs,void* modelMemory,void* correctInput);

#endif // INCLUDED_VERSAT_AI
