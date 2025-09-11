/*
 * SPDX-FileCopyrightText: 2025 IObundle
 *
 * SPDX-License-Identifier: MIT
 */

#include "iob_bsp.h"
#include "iob_printf.h"
#include "iob_timer.h"
#include "iob_uart.h"
#include "versat_accel.h"
#include "versat_ai_conf.h"
#include "versat_ai_mmap.h"
#include <string.h>

#include "modelInfo.h"
#include "versat_ai.h"

char *send_string = "Sending this string as a file to console.\n"
                    "The file is then requested back from console.\n"
                    "The sent file is compared to the received file to confirm "
                    "correct file transfer via UART using console.\n"
                    "Generating the file in the firmware creates an uniform "
                    "file transfer between pc-emul, simulation and fpga without"
                    " adding extra targets for file generation.\n";

void clear_cache() {
#ifndef PC
  // Delay to ensure all data is written to memory
  printf("Gonna clear the cache\n");
  for (unsigned int i = 0; i < 10; i++)
    asm volatile("nop");
  // Flush VexRiscv CPU internal cache
  asm volatile(".word 0x500F" ::: "memory");
#endif
}

int main() {
  char pass_string[] = "Test passed!";
  char fail_string[] = "Test failed!";

  // init timer
  timer_init(TIMER0_BASE);

  // init uart
  uart_init(UART0_BASE, IOB_BSP_FREQ / IOB_BSP_BAUD);
  printf_init(&uart_putc);

  // test puts
  uart_puts("\n\n\nHello world from versat_ai!\n\n\n");

  uart_puts("\n\n\nGonna init versat!\n\n\n");
  SetVersatDebugPrintfFunction(printf);
  versat_init(VERSAT0_BASE);

  ConfigCreateVCD(false);

  printf("Versat base: %x\n", VERSAT0_BASE);

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
  uart_recvfile("inputs.bin", inputMemory);

  DebugRunInference(output, temp, inputs, model, correct);

  uart_sendfile("test.log", strlen(pass_string), pass_string);

  free(output);
  free(temp);
  free(model);
  free(correct);
  free(inputMemory);

  // read current timer count, compute elapsed time
  unsigned long long elapsed = timer_get_count();
  unsigned int elapsedu = elapsed / (IOB_BSP_FREQ / 1000000);

  printf("\nExecution time: %d clock cycles\n", (unsigned int)elapsed);
  printf("\nExecution time: %dus @%dMHz\n\n", elapsedu, IOB_BSP_FREQ / 1000000);

  uart_finish();
}
