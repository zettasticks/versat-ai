#include "stdio.h"
#include "stdlib.h"

#include "software/versat_ai.h"

#define OUTPUT_SIZE 40
#define TEMP_SIZE 60416
#define MODEL_SIZE 24008
#define CORRECT_SIZE 131536

int main(int argc,const char* argv[]){
  void* output = malloc(OUTPUT_SIZE);
  void* temp = malloc(TEMP_SIZE);
  void* model = malloc(MODEL_SIZE);
  void* correct = malloc(CORRECT_SIZE);

  void* inputs[1];
  inputs[0] = NULL;
  
  FILE* modelFile = fopen("scripts/model.bin","rb");
  if(!modelFile){
    printf("Error opening model\n");
  }

  size_t readded = fread(model,sizeof(char),MODEL_SIZE,modelFile);
  if(readded != MODEL_SIZE){
    printf("Error reading model\n");
  }

  FILE* correctFile = fopen("scripts/correctOutput.bin","rb");
  if(!correctFile){
    printf("Error opening correct\n");
  }

  readded = fread(correct,sizeof(char),CORRECT_SIZE,correctFile);
  if(readded != CORRECT_SIZE){
    printf("Error reading correct\n");
  }

  printf("here\n");
  
  InferenceOutput out = DebugRunInference(output,temp,inputs,model,correct);

  return 0;
}
