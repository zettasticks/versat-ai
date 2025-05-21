#include "stdio.h"
#include "stdlib.h"

#include "software/versat_ai.h"

int main(int argc,const char* argv[]){
  void* output = malloc(GetOutputMemorySize());
  void* temp = malloc(GetTemporaryMemorySize());
  void* model = malloc(GetModelMemorySize());
  void* correct = malloc(GetCorrectMemorySize());

  void* inputs[1];

  printf("Output : %p\n",output);
  printf("Temp   : %p\n",temp);
  printf("Model  : %p\n",model);
  printf("Correct: %p\n",correct);

  inputs[0] = malloc(100000);
  
  FILE* modelFile = fopen("model.bin","rb");
  if(!modelFile){
    printf("Error opening model\n");
  }

  size_t readded = fread(model,sizeof(char),GetModelMemorySize(),modelFile);
  if(readded != GetModelMemorySize()){
    printf("Error reading model\n");
  }

  FILE* correctFile = fopen("correctOutput.bin","rb");
  if(!correctFile){
    printf("Error opening correct\n");
  }

  readded = fread(correct,sizeof(char),GetCorrectMemorySize(),correctFile);
  if(readded != GetCorrectMemorySize()){
    printf("Error reading correct\n");
  }
  
  /* InferenceOutput out = */ DebugRunInference(output,temp,inputs,model,correct);

  // Mainly for address sanitizer to not complain
  free(output);
  free(temp);
  free(model);
  free(correct);
  free(inputs[0]);

  return 0;
}
