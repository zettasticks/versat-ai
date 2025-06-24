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
  versat_init(VERSAT0_BASE);

  printf("here\n");

  int *testMem = (int *)malloc(4);
  *testMem = 1234;
  clear_cache();

  accelConfig->debug.address = testMem;

  printf("Address: %p\n", testMem);
  printf("Value Before = %d\n\n", accelState->debug.lastRead);

  RunAccelerator(3);

  printf("Value After = %d\n\n", accelState->debug.lastRead);

  // test printf with floats
  printf("Value of Pi = %f\n\n", 3.1415);

// Don't transfer files when running alongside tester
#ifndef TESTER
  // test file send
  char *sendfile = malloc(1000);
  int send_file_size = 0;
  send_file_size = strlen(strcpy(sendfile, send_string));
  uart_sendfile("Sendfile.txt", send_file_size, sendfile);

  // test file receive
  char *recvfile = malloc(10000);
  int file_size = 0;
  file_size = uart_recvfile("Sendfile.txt", recvfile);

  // compare files
  if (strcmp(sendfile, recvfile)) {
    printf("FAILURE: Send and received file differ!\n");
  } else {
    printf("SUCCESS: Send and received file match!\n");
  }

  free(sendfile);
  free(recvfile);

  uart_sendfile("test.log", strlen(pass_string), pass_string);
#endif // TESTER

  // read current timer count, compute elapsed time
  unsigned long long elapsed = timer_get_count();
  unsigned int elapsedu = elapsed / (IOB_BSP_FREQ / 1000000);

  printf("\nExecution time: %d clock cycles\n", (unsigned int)elapsed);
  printf("\nExecution time: %dus @%dMHz\n\n", elapsedu, IOB_BSP_FREQ / 1000000);

  uart_finish();
}
