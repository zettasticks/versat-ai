#ifndef INCLUDED_VERSAT_AI
#define INCLUDED_VERSAT_AI

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

// Memory is allocated by the user.

// TODO: This should not be here, we do not want to polute user space with our
// code.
#define OFFSET_PTR(PTR, OFFSET) ((void *)(((char *)PTR) + OFFSET))

// TODO: Remove this.
// What we should have is a function that goes:
// OutputInfo GetOutput(void* output,int index);
// That not only fixes the output to the correct address but also fills it with
// some information.

typedef struct {
  int numberOfOutputs;
  void **outputLocation; // outputLocation[N] gives the address of the N output.
} InferenceOutput;

typedef struct {
  const char *name;
  const char *typeName;
  size_t outputSize;
} LayerInfo;

typedef struct {
  // Extra info to help
  int maxDims;
  int64_t *firstInputDim;
  int64_t *secondInputDim;
  int64_t *broadCastedShape;
} AddInfo;

typedef struct {
  int dims;
  int64_t *inputDims;
} ReluInfo;

typedef struct {
  int dims;
  int64_t *inputDims;
  int64_t *outputDims;
  int kernelSizeW;
  int kernelSizeH;
} MaxPoolInfo;

typedef struct {
#if 0
  // Attributes as described by onnx.
  int auto_pad;
  int* dilations;
  int group;
  int* kernel_shape;
  int* pads;
  int* strides;
#endif

  // Extra info to help
  int numberInputDims;
  int64_t *inputDims;
  int numberKernelDims;
  int64_t *kernelDims;
  int numberOutDims;
  int64_t *outDims;
} ConvInfo;

typedef struct {
  int64_t *inputDims;
  int numberInputDims;
  int numberShapeDims;
} ReshapeInfo;

typedef struct {
  int64_t *inputADims;
  int numberInputADims;
  int64_t *inputBDims;
  int numberInputBDims;
  int64_t *outputDims;
  int numberOutputDims;
} MatMulInfo;

extern LayerInfo layers[];
extern int numberLayers;

// Operations
void *Conv(void *inputX, void *inputW, void *output, int index, ConvInfo *info);
void *Reshape(void *data, void *shape, void *output, int index,
              ReshapeInfo *info);
void *Add(void *inputA, void *inputB, void *output, int index, AddInfo *info);
void *Relu(void *inputX, void *output, int index, ReluInfo *info);
void *MaxPool(void *inputX, void *output, int index, MaxPoolInfo *info);
void *MatMul(void *inputA, void *inputB, void *output, int index,
             MatMulInfo *info);
void AssertAlmostEqual(void *toTest, void *correctValues, int index);

// Output and Temporary memory must be allocated by the user. Call the
// GetXXXMemorySize functions to obtain the amount of memory required. Model
// memory is memory that contains a loaded model (provided by a binary file that
// must be read at runtime).
InferenceOutput RunInference(void *outputMemory, void *temporaryMemory,
                             void **inputs, void *modelMemory);
InferenceOutput DebugRunInference(void *outputMemory, void *temporaryMemory,
                                  void **inputs, void *modelMemory,
                                  void *correctInput);

#endif // INCLUDED_VERSAT_AI
