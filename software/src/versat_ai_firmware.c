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

#include "versat_ai.h"

// Contains a set of defines for each test type.
#include "testInfo.h"

#ifndef ARRAY_SIZE
#define ARRAY_SIZE(ARR) ((sizeof(ARR) / sizeof(ARR[0])))
#endif

#ifdef PC
#include <stdio.h>
#include <unistd.h> // for sleep()
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

void FastReceiveFile(const char *pathPrefix, const char *path, void *buffer,
                     int expectedSize) {
  char fullPath[128];
  snprintf(fullPath, 128, "%s_%s", pathPrefix, path);

#ifdef PC
  FILE *f = fopen(fullPath, "r");
  if (!f) {
    printf("Problem opening file for reading: %s\n", fullPath);
    return;
  }

  long int size = GetFileSize(f);
  fread(buffer, sizeof(char), size, f);
  fclose(f);
#else
  ethernet_receive_file(fullPath, buffer, expectedSize);
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

  ConfigCreateVCD(true);

  printf("Versat base: %x\n", VERSAT0_BASE);

  int stackVar;

  printf("Stack  : %p\n", &stackVar);

  // We allocate a little bit more just in case.
  // Also need to allocate a bit more to ensure that Align4 works fine.
  int extra = 16;

  for (int i = 0; i < ARRAY_SIZE(testModels); i++) {
    TestModelInfo info = *testModels[i];

    printf("\n\n");
    printf("\n==============================\n");
    printf("Gonna run the full test named: %s", info.namespace);
    printf("\n==============================\n");

    // TODO: Arena stuff, using malloc so much is starting to scare me.

    void *output = Align4(malloc(info.outputSize + extra));
    void *temp = Align4(malloc(info.tempSize + extra));
    void *model = Align4(malloc(info.modelSize + extra));
    void *correct = Align4(malloc(info.correctSize + extra));
    void *inputMemory = Align4(malloc(info.totalInputSize + extra));

    *((int *)output) = 123;
    *((int *)temp) = 123;
    *((int *)model) = 123;
    *((int *)correct) = 123;
    *((int *)inputMemory) = 123;

    void **inputs = Align4(malloc(sizeof(void *) * info.inputCount));
    for (int i = 0; i < info.inputCount; i++) {
      inputs[i] = OFFSET_PTR(inputMemory, info.inputOffsets[i]);
    }

    printf("Output : %p\n", output);
    printf("Temp   : %p\n", temp);
    printf("Model  : %p\n", model);
    printf("Correct: %p\n", correct);
    printf("Input  : %p\n", inputMemory);

    void *total = inputs[info.inputCount - 1];
    if (info.inputCount == 0) {
      total = OFFSET_PTR(correct, info.correctSize);
    }
    printf("Total  : %p\n", total);

    if (total > &stackVar) {
      printf(
          "Error, we run out of memory, increase the value of firm_w argument "
          "and setup again\n");
      uart_finish();
      return 0;
    }

    FastReceiveFile(info.namespace, "correctOutputs.bin", correct,
                    info.correctSize);
    printf("Received correct outputs\n");
    FastReceiveFile(info.namespace, "model.bin", model, info.modelSize);
    printf("Received model\n");
    FastReceiveFile(info.namespace, "inputs.bin", inputMemory,
                    info.totalInputSize);
    printf("Received inputs\n");

    info.debugInferenceFunction(output, temp, inputs, model, correct);

    free(output);
    free(temp);
    free(model);
    free(correct);
    free(inputMemory);
  }

#ifdef PC
  sleep(1);
#endif

  uart_sendfile("test.log", strlen(pass_string), pass_string);

  // read current timer count, compute elapsed time
  unsigned long long elapsed = timer_get_count();
  unsigned int elapsedu = elapsed / (IOB_BSP_FREQ / 1000000);

  printf("\nExecution time: %d clock cycles\n", (unsigned int)elapsed);
  printf("\nExecution time: %dus @%dMHz\n\n", elapsedu, IOB_BSP_FREQ / 1000000);

  uart_finish();

  return 0;
}
