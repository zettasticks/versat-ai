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

#ifndef PC
#include "iob_eth.h"
#endif

#include "modelInfo.h"
#include "versat_ai.h"

// Contains a set of defines for each test type.
#include "testInfo.h"

#ifdef PC
#include <stdio.h>
long int GetFileSize(FILE *file) {
  long int mark = ftell(file);

  fseek(file, 0, SEEK_END);
  long int size = ftell(file);

  fseek(file, mark, SEEK_SET);

  return size;
}
#endif

#ifndef PC
uint32_t uart_request_ethernet_recvfile(const char *file_name) {
  uart_puts(UART_PROGNAME);
  uart_puts(": requesting to receive file by ethernet\n");

  // send file receive by ethernet request
  uart_putc(0x13);

  // send file name (including end of string)
  uart_puts(file_name);
  uart_putc(0);

  // receive file size
  uint32_t file_size = uart_getc();
  file_size |= ((uint32_t)uart_getc()) << 8;
  file_size |= ((uint32_t)uart_getc()) << 16;
  file_size |= ((uint32_t)uart_getc()) << 24;

  // send ACK before receiving file
  uart_putc(ACK);

  return file_size;
}

void ethernet_receive_file(const char *path, void *buffer, int expectedSize) {
  if (expectedSize == 0) {
    return;
  }
  uint32_t size = uart_request_ethernet_recvfile(path);
  eth_rcv_file(buffer, size);
}
#endif

void FastReceiveFile(const char *path, void *buffer, int expectedSize) {
#ifdef PC
  FILE *f = fopen(path, "r");
  if (!f) {
    printf("Problem opening file for reading: %s\n", path);
    return;
  }

  long int size = GetFileSize(f);
  fread(buffer, sizeof(char), size, f);
  fclose(f);
#else
  ethernet_receive_file(path, buffer, expectedSize);
#endif
}

void silent_clear_cache() {
#ifndef PC
  for (unsigned int i = 0; i < 10; i++)
    asm volatile("nop");
  // Flush VexRiscv CPU internal cache
  asm volatile(".word 0x500F" ::: "memory");
#endif
}

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

void *Align4(void *in) {
  iptr asInt = (iptr)in;

  asInt = ((asInt + 3) & ~3);
  return (void *)asInt;
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

#ifdef TEST_NAME
  printf("\n\nRunning test %s\n\n", TEST_NAME);
#endif

#ifndef PC
  uart_puts("\nGonna init ethernet\n");
  eth_init(ETH0_BASE, &silent_clear_cache);
  eth_wait_phy_rst();
#endif

  uart_puts("\nGonna init versat!\n");
  SetVersatDebugPrintfFunction(printf);
  versat_init(VERSAT0_BASE);

  ConfigCreateVCD(false);

  printf("Versat base: %x\n", VERSAT0_BASE);

  int stackVar;

  printf("Stack  : %p\n", &stackVar);

  // We allocate a little bit more just in case.
  // Also need to allocate a bit more to ensure that Align4 works fine.
  int extra = 100;

  void *output = Align4(malloc(VERSAT_AI_OUTPUT_SIZE + extra));
  void *temp = Align4(malloc(VERSAT_AI_TEMP_SIZE + extra));
  void *model = Align4(malloc(VERSAT_AI_MODEL_SIZE + extra));
  void *correct = Align4(malloc(VERSAT_AI_CORRECT_SIZE + extra));
  void *inputMemory = Align4(malloc(VERSAT_AI_ALL_INPUTS_SIZE + extra));

  void *inputs[VERSAT_AI_N_INPUTS];
  for (int i = 0; i < VERSAT_AI_N_INPUTS; i++) {
    inputs[i] = OFFSET_PTR(inputMemory, VERSAT_AI_INPUT_OFFSET[i]);
  }

  printf("Output : %p\n", output);
  printf("Temp   : %p\n", temp);
  printf("Model  : %p\n", model);
  printf("Correct: %p\n", correct);
  printf("Input  : %p\n", inputMemory);

  void *total = ((void *)((char *)inputMemory) + VERSAT_AI_ALL_INPUTS_SIZE);
  printf("Total  : %p\n", total);

  if (total > &stackVar) {
    printf("Error, we run out of memory, increase the value of firm_w argument "
           "and setup again\n");
    uart_finish();
    return 0;
  }

  FastReceiveFile("correctOutputs.bin", correct, VERSAT_AI_OUTPUT_SIZE);
  printf("Received correct outputs\n");
  FastReceiveFile("model.bin", model, VERSAT_AI_MODEL_SIZE);
  printf("Received model\n");
  FastReceiveFile("inputs.bin", inputMemory, VERSAT_AI_ALL_INPUTS_SIZE);
  printf("Received inputs\n");

  DebugRunInference(output, temp, inputs, model, correct);

#ifdef PC
  sleep(1);
#endif

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

  return 0;
}
