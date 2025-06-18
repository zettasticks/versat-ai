/*
 * SPDX-FileCopyrightText: 2025 IObundle
 *
 * SPDX-License-Identifier: MIT
 */

int main() { return 0; }

#if 0

#include "iob_bsp.h"
#include "iob_printf.h"
#include "iob_timer.h"
#include "iob_uart.h"

#include "versat_ai_conf.h"
#include "versat_ai_mmap.h"

#include "modelInfo.h"
#include "versat_ai.h"

#define OFFSET_PTR(PTR, OFFSET) ((void *)(((char *)PTR) + OFFSET))

#define FREQ 100000000
#define BAUD 3000000

int main() {
  unsigned long long elapsed;
  unsigned int elapsedu;

  // init timer and uart
  timer_init(TIMER0_BASE);
  uart_init(UART0_BASE, IOB_BSP_FREQ / IOB_BSP_BAUD);
  printf_init(&uart_putc);

  printf("\nHello world!\n");
  printf("Running Versat AI!\n");

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

  uart_recvfile("model.bin", model);
  uart_recvfile("correctOutputs.bin", correct);
  uart_recvfile("inputs.bin", inputs);

#if 0
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
#endif

  DebugRunInference(output, temp, inputs, model, correct);

  // Mainly for address sanitizer to not complain
  free(output);
  free(temp);
  free(model);
  free(correct);
  free(inputs[0]);

  return 0;

  uart_finish();
}

#endif
