#ifndef INCLUDED_VERSAT_AI
#define INCLUDED_VERSAT_AI

#include <stddef.h>

// Memory is allocated by the user. 

// These functions are constant, meaning that buffers can be reused if needed 
size_t GetModelSize();
size_t GetOutputMemorySize();
size_t GetTemporaryMemorySize();

typedef struct{
   int numberOfOutputs;
   void** outputLocation; // outputLocation[N] gives the address of the N output.
} InferenceOutput;

// Output and Temporary memory must be allocated by the user. Call the GetXXXMemorySize functions to obtain the amount of memory required.
// Model memory is memory that contains a loaded model (provided by a binary file that must be read at runtime).
InferenceOutput RunInference(void* outputMemory,void* temporaryMemory,void** inputs,void* modelMemory);
InferenceOutput DebugRunInference(void* outputMemory,void* temporaryMemory,void** inputs,void* modelMemory,void* correctInput);

#endif // INCLUDED_VERSAT_AI
