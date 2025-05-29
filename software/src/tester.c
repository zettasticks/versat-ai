#if 0

#include "stdio.h"
#include "stdlib.h"

#include "output/modelInfo.h"
#include "software/versat_ai.h"

#define OFFSET_PTR(PTR, OFFSET) ((void *)(((char *)PTR) + OFFSET))

int main(int argc, const char *argv[]) {
  void *output = malloc(VERSAT_AI_OUTPUT_SIZE);
  void *temp = malloc(VERSAT_AI_TEMP_SIZE);
  void *model = malloc(VERSAT_AI_MODEL_SIZE);
  void *correct = malloc(VERSAT_AI_CORRECT_SIZE);
  void *inputMemory = malloc(VERSAT_AI_ALL_INPUTS_SIZE);

  void *inputs[VERSAT_AI_N_INPUTS];
  for (int i = 0; i < VERSAT_AI_N_INPUTS; i++) {
    inputs[i] = OFFSET_PTR(inputMemory, VERSAT_AI_INPUT_OFFSET[i]);
  }

  printf("Output : %p\n", output);
  printf("Temp   : %p\n", temp);
  printf("Model  : %p\n", model);
  printf("Correct: %p\n", correct);

  FILE *modelFile = fopen("model.bin", "rb");
  if (!modelFile) {
    printf("Error opening model\n");
  }

  size_t readded = fread(model, sizeof(char), VERSAT_AI_MODEL_SIZE, modelFile);
  if (readded != VERSAT_AI_MODEL_SIZE) {
    printf("Error reading model\n");
  }

  FILE *correctFile = fopen("correctOutputs.bin", "rb");
  if (!correctFile) {
    printf("Error opening correct\n");
  }

  readded = fread(correct, sizeof(char), VERSAT_AI_CORRECT_SIZE, correctFile);
  if (readded != VERSAT_AI_CORRECT_SIZE) {
    printf("Error reading correct\n");
  }

  FILE *inputFile = fopen("inputs.bin", "rb");
  if (!inputFile) {
    printf("Error opening inputs\n");
  }

  readded =
      fread(inputMemory, sizeof(char), VERSAT_AI_ALL_INPUTS_SIZE, inputFile);
  if (readded != VERSAT_AI_ALL_INPUTS_SIZE) {
    printf("Error reading inputs\n");
  }

  /* InferenceOutput out = */ DebugRunInference(output, temp, inputs, model,
                                                correct);

  // Mainly for address sanitizer to not complain
  free(output);
  free(temp);
  free(model);
  free(correct);
  free(inputs[0]);

  return 0;
}

#endif
