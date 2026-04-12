/*
 * SPDX-FileCopyrightText: 2025 IObundle
 *
 * SPDX-License-Identifier: MIT
 */

#if 0

#include "iob_bsp.h"
//#include "iob_gpio_csrs.h"
#include "iob_printf.h"
#include "iob_system_tester_conf.h"
#include "iob_system_tester_mmap.h"
#include "iob_timer.h"
#include "iob_uart.h"
#include <string.h>
#ifdef IOB_SYSTEM_TESTER_USE_ETHERNET
#include "iob_eth.h"
#endif

#include "iob_regfileif_csrs.h"

// Enable debug messages.
#define DEBUG 0

#ifdef IOB_SYSTEM_TESTER_USE_ETHERNET

// Ethernet utility functions
void clear_cache() {
  // Delay to ensure all data is written to memory
  for (unsigned int i = 0; i < 10; i++)
    asm volatile("nop");
  // Flush VexRiscv CPU internal cache
  asm volatile(".word 0x500F" ::: "memory");
}

// Send signal by uart to receive file by ethernet
uint32_t uart_recvfile_ethernet(const char *file_name) {

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
#endif // IOB_SYSTEM_TESTER_USE_ETHERNET

int main() {
  int i;
  uint32_t file_size = 0;
  char c, buffer[2048];
  char pass_string[] = "Test passed!";
  char fail_string[] = "Test failed!";

  // init timer
  timer_init(TIMER0_BASE);

  // init uart
  uart_init(UART0_BASE, IOB_BSP_FREQ / IOB_BSP_BAUD);
  printf_init(&uart_putc);

#ifdef IOB_SYSTEM_TESTER_USE_ETHERNET
  // Init ethernet
  eth_init(ETH0_BASE, &clear_cache);
  // Wait for PHY reset to finish
  eth_wait_phy_rst();
#endif // IOB_SYSTEM_TESTER_USE_ETHERNET

  // // init gpio
  // iob_gpio_csrs_init_baseaddr(GPIO0_BASE);
  // // Use GPIO output as timing flags for debug with VCD
  // iob_gpio_csrs_set_output_0(0x0);

  // test puts
  uart_puts("\n\n\nHello world from Tester!\n\n\n");

  // #ifdef IOB_SYSTEM_TESTER_USE_ETHERNET
  //   // Receive data from console via Ethernet
  //   file_size = uart_recvfile_ethernet(
  //       "xcelium_cov.tcl"); // NOTE: random file just for demo
  //   eth_rcv_file(buffer, file_size);
  //   uart_puts("\nFile received from console via ethernet:\n");
  //   for (i = 0; i < file_size; i++)
  //     uart_putc(buffer[i]);
  // #endif // IOB_SYSTEM_TESTER_USE_ETHERNET

  //
  // Init SUT
  //

  // Init SUT (connected through REGFILEIF)
  iob_regfileif_csrs_init_baseaddr(SUT_BASE);

  // Place SUT into reset state
  iob_regfileif_csrs_set_rst(1);

  // Load SUT firmware into SUT's memory zone (external memory)
#ifdef IOB_SYSTEM_TESTER_USE_ETHERNET
  // Receive data from console via Ethernet
  file_size =
      uart_recvfile_ethernet("../../../software/versat_ai_firmware.bin");
  eth_rcv_file((char *)0x40000000,
               file_size); // Place SUT fw in external memory
#else  // NOT IOB_SYSTEM_TESTER_USE_ETHERNET
  // Receive data from console via UART
  file_size =
      uart_recvfile("../../../software/versat_ai_firmware.bin",
                    (char *)0x40000000); // Place SUT fw in external memory
#endif // IOB_SYSTEM_TESTER_USE_ETHERNET

  uart_puts("[Tester]: Initializing SUT via UART...\n");

  // Disable SUT reset
  iob_regfileif_csrs_set_rst(0);

  // Init and switch to uart1 (connected to the SUT)
  uart_init(UART1_BASE, IOB_BSP_FREQ / IOB_BSP_BAUD);

  // Wait for ENQ signal from SUT
  while ((c = uart_getc()) != ENQ)
    if (DEBUG) {
      iob_uart_csrs_init_baseaddr(UART0_BASE);
      uart_putc(c);
      iob_uart_csrs_init_baseaddr(UART1_BASE);
    };

  // Send ack to sut
  uart_puts("\nTester ACK");

  iob_uart_csrs_init_baseaddr(UART0_BASE);
  uart_puts("[Tester]: Received SUT UART enquiry and sent acknowledge.\n");

  //
  // Read SUT messages
  //

  uart_puts("\n[Tester]: Reading SUT messages...\n");
  iob_uart_csrs_init_baseaddr(UART1_BASE);

  // Delay to ensure SUT is waiting for ack
  for (unsigned int i = 0; i < 100; i++)
    asm volatile("nop");
  // Send second ack to SUT to continue boot
  uart_putc(ACK);

  i = 0;
  // Read and store messages sent from SUT in a buffer to later be printed
  // while ((c = uart_getc()) != EOT) {
  //   buffer[i] = c;
  //   if (DEBUG) {
  //     iob_uart_csrs_init_baseaddr(UART0_BASE);
  //     uart_putc(c);
  //     iob_uart_csrs_init_baseaddr(UART1_BASE);
  //   }
  //   i++;
  // }
  // buffer[i] = EOT;

  // Alternative: Print characters received from SUT as soon as they arrive
  // This alternative is better to see satus of SUT in real time, but tester may
  // miss/skip some characters if it cant read them fast enough. One solution to
  // avoid skipping characters is using a UART that includes a FIFO (like
  // uart16550).
  while ((c = uart_getc()) != EOT) {
    iob_uart_csrs_init_baseaddr(UART0_BASE);
    uart_putc(c);
    iob_uart_csrs_init_baseaddr(UART1_BASE);
  }

  //
  // Print (stored) SUT messages
  //

  // Switch back to UART0
  iob_uart_csrs_init_baseaddr(UART0_BASE);

  // // Print messages previously stored from SUT in the buffer
  // uart_puts("[Tester]: #### Messages received from SUT: ####\n\n");
  // if (!DEBUG) {
  //   for (i = 0; buffer[i] != EOT; i++) {
  //     uart_putc(buffer[i]);
  //   }
  // }
  // uart_puts("\n[Tester]: #### End of messages received from SUT ####\n\n");

  //
  // End test
  //

  uart_sendfile("test.log", strlen(pass_string), pass_string);

  uart_finish();
}

#endif

/* Includes */
#include "iob_bsp.h"
#include "iob_printf.h"
#include "iob_system_tester_conf.h"
#include "iob_system_tester_mmap.h"
#include "iob_timer.h"
#include "iob_uart.h"
#include <string.h>

#include "iob_regfileif_csrs.h"
#include "versat_ai.h"
#include <stdbool.h>

void clear_cache() {
  // Delay to ensure all data is written to memory
  for (unsigned int i = 0; i < 10; i++)
    asm volatile("nop");

  // Flush VexRiscv CPU internal cache
  asm volatile(".word 0x500F" ::: "memory");
}

#ifdef USE_ETHERNET
#include "iob_eth.h"

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
  if (expectedSize == 0) {
    return;
  }

  char fullPath[128];
  snprintf(fullPath, 128, "%s_%s", pathPrefix, path);

#if PC
  FILE *f = fopen(fullPath, "r");
  if (!f) {
    printf("Problem opening file for reading: %s\n", fullPath);
    return;
  }

  long int size = GetFileSize(f);
  fread(buffer, sizeof(char), size, f);
  fclose(f);
  return;
#endif

#if USE_ETHERNET
  ethernet_receive_file(fullPath, buffer, expectedSize);
#else
  uart_recvfile(fullPath, buffer);
  printf("Received file by uart\n");
#endif
}

/*
  Remember:
  Tester contains an internal memory and that is where the firmware is running
  from. 0x00000000 to 0x40000000 is internal memory. 0x40000000 to 0x80000000 is
  external memory and that is the same view that the sut contains. Therefore
    Tester 0x4YYYYYYY - 0x7YYYYYYY is the same as SUT 0x0YYYYYYY - 0x3YYYYYYY

  However the SUT is also offseted by 0x10000000 meaning that we need to offset
  pointers by 0x50000000
*/

// Receives file into mem
uint32_t uart_filesize(char *file_name) {
  uart_puts(UART_PROGNAME);
  uart_puts(": requesting to get file size\n");

  // send file receive request
  uart_putc(0x09);

  // send file name
  uart_sendstr(file_name);

  // receive file size
  uint32_t file_size = uart_getc();
  file_size |= ((uint32_t)uart_getc()) << 8;
  file_size |= ((uint32_t)uart_getc()) << 16;
  file_size |= ((uint32_t)uart_getc()) << 24;

  uart_putc(ACK);

  return file_size;
}

typedef struct {
  void *data;
  uint32_t size;
} File;

File GetFile(const char *path) {
  uint32_t size = uart_filesize(path);
  void *data = malloc(size + 16);
  uart_recvfile(path, data);

  File res = {};
  res.data = data;
  res.size = size;

  return res;
}

File GetFileNoAlloc(const char *path, void *buffer) {
  uint32_t size = uart_filesize(path);

#if USE_ETHERNET
  ethernet_receive_file(path, buffer, size);
#else
  uart_recvfile(path, buffer);
#endif

  File res = {};
  res.data = buffer;
  res.size = size;

  return res;
}

static bool IsAlpha(char ch) {
  bool res = false;

  res |= (ch >= 'a' && ch <= 'z');
  res |= (ch >= 'A' && ch <= 'Z');
  res |= (ch >= '0' && ch <= '9');
  res |= (ch == '_');

  return res;
}

#ifdef IOB_SYSTEM_TESTER_USE_ETHERNET

// Send signal by uart to receive file by ethernet
uint32_t uart_recvfile_ethernet(const char *file_name) {

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
#endif // IOB_SYSTEM_TESTER_USE_ETHERNET

void relay_messages() {
  uint8_t c = 0;

  iob_uart_csrs_init_baseaddr(UART1_BASE);
  if (iob_uart_csrs_get_rxready()) {
    c = uart_getc();
  }
  iob_uart_csrs_init_baseaddr(UART0_BASE);
  if (c != 0) {
    uart_putc(c);
  }
}

void clear_sut_messages() {
  char ch = 0;
  for (int i = 0; i < 10000; i++) {
    iob_uart_csrs_init_baseaddr(UART1_BASE);
    if (!iob_uart_csrs_get_rxready()) {
      break;
    }
    ch = uart_getc();
    iob_uart_csrs_init_baseaddr(UART0_BASE);
    uart_putc(ch);
  }
  iob_uart_csrs_init_baseaddr(UART0_BASE);
}

volatile char *eth_frame_ptr = NULL;

int main() {
  int i;
  uint32_t file_size = 0;
  char c, buffer[2048];
  char pass_string[] = "Test passed!";
  char fail_string[] = "Test failed!";

  // init timer
  timer_init(TIMER0_BASE);

  // Init SUT UART
  uart_init(UART1_BASE, IOB_BSP_FREQ / IOB_BSP_BAUD);

  // Init uart0 (Outside communication)
  uart_init(UART0_BASE, IOB_BSP_FREQ / IOB_BSP_BAUD);
  printf_init(&uart_putc);

  // Init uart1 (connected to the SUT)
  // uart_init(UART1_BASE, IOB_BSP_FREQ / IOB_BSP_BAUD);

  // Allocate the last portion for the ethernet.
  eth_frame_ptr = (volatile char *)0x5fff0000;

#ifdef IOB_SYSTEM_TESTER_USE_ETHERNET
  // Init ethernet
  eth_init(ETH0_BASE, &clear_cache);
  // Wait for PHY reset to finish
  eth_wait_phy_rst();
#endif // IOB_SYSTEM_TESTER_USE_ETHERNET

  uart_puts("\n\n\nHello world from Tester!\n\n\n");

  // Inits sut.
  // Init SUT (connected through REGFILEIF)
  iob_regfileif_csrs_init_baseaddr(SUT_BASE);

  // Place SUT into reset state
  iob_regfileif_csrs_set_rst(1);

  // Load SUT firmware into SUT's memory zone (external memory)
#ifdef IOB_SYSTEM_TESTER_USE_ETHERNET
  // Receive data from console via Ethernet
  file_size = uart_recvfile_ethernet("../software/versat_ai_firmware.bin");
  eth_rcv_file((char *)0x40000000,
               file_size); // Place SUT fw in external memory
#else                      // NOT IOB_SYSTEM_TESTER_USE_ETHERNET
  // Receive data from console via UART
  file_size =
      uart_recvfile("../software/versat_ai_firmware.bin",
                    (char *)0x40000000); // Place SUT fw in external memory
#endif                     // IOB_SYSTEM_TESTER_USE_ETHERNET

  uart_puts("[Tester]: Initializing SUT via UART...\n");

  // Disable SUT reset
  iob_regfileif_csrs_set_rst(0);

#if 1

  // Wait for ENQ signal from SUT
  iob_uart_csrs_init_baseaddr(UART1_BASE);
  while ((c = uart_getc()) != ENQ)
    if (DEBUG) {
      iob_uart_csrs_init_baseaddr(UART0_BASE);
      uart_putc(c);
      iob_uart_csrs_init_baseaddr(UART1_BASE);
    };

  // Send ack to SUT to continue boot
  uart_putc(ACK);

  // NOTE: Does this need to be after or before the ENQ signal loop?
  // iob_regfileif_csrs_set_start(1);

  // Send ack to sut
  // uart_puts("\nTester ACK");

  iob_uart_csrs_init_baseaddr(UART0_BASE);
  uart_puts("[Tester]: Received SUT UART enquiry and sent acknowledge.\n");

  //
  // Read SUT messages
  //

  // Allows SUT to progress the boot.
  iob_regfileif_csrs_set_start(0);
  // uart_putc(ACK);

  // At this point the SUT is running freely

  uart_puts("\n[Tester]: Reading SUT messages...\n");
  iob_uart_csrs_init_baseaddr(UART1_BASE);

  // Delay to ensure SUT is waiting for ack
  for (unsigned int i = 0; i < 100; i++)
    asm volatile("nop");

  iob_regfileif_csrs_set_start(0);

  // At this point the SUT is running, right?

  clear_sut_messages();

  printf("Cleared SUT messages\n");

#endif

  int *malloced = (int *)malloc(sizeof(int));

  printf("Tester Malloc gave pointer: %p\n", malloced);

  // Need to make sure that we do not overwrite code or stack.
  // The first 256 bytes are reserved for passing values around.
  // char *sutMemoryBase = (char *)0x41000100;

  int32_t sutMemoryOffset = 0x40000000;
  int32_t transferOffset = 0x02000000;

  char *sutMemoryBase = (char *)(sutMemoryOffset + transferOffset + 0x00000100);

  File metadata = GetFile("VERSAT_TEST_METADATA.txt");

  char *ptr = (char *)metadata.data;
  char *end = ptr + metadata.size;

  for (; ptr < end;) {
    if (!IsAlpha(*ptr)) {
      break;
    }

    char *lineStart = ptr;
    while (IsAlpha(*ptr)) {
      ptr += 1;
    }
    char *lineEnd = ptr;
    int size = lineEnd - lineStart;

    ptr = lineEnd + 1;

    char pathBuffer[256];
    sprintf(pathBuffer, "%.*s_metamodel.bin", size, lineStart);

    File metamodel = GetFileNoAlloc(pathBuffer, sutMemoryBase);
    CompiledModel *compiledModel = (CompiledModel *)metamodel.data;

    printf("Output: %d\n", compiledModel->outputSize);
    printf("Temp: %d\n", compiledModel->tempSize);
    printf("Model: %d\n", compiledModel->modelSize);
    printf("Correct: %d\n", compiledModel->correctSize);
    printf("Input: %d\n", compiledModel->totalInputSize);

    // Allocate space for each memory buffer, +16 to give us some wiggle room.
    // Proper code should work without this but we will handle this later.
    char *output = sutMemoryBase + metamodel.size + 16;
    char *temp = output + compiledModel->outputSize + 16;
    char *model = temp + compiledModel->tempSize + 16;
    char *correct = model + compiledModel->modelSize + 16;
    char *inputs = correct + compiledModel->correctSize + 16;

    printf("Output: %p\n", output);
    printf("Temp: %p\n", temp);
    printf("Model: %p\n", model);
    printf("Correct: %p\n", correct);
    printf("Input: %p\n", inputs);

    void **inputsVector = inputs + compiledModel->totalInputSize;
    uint32_t *inputOffsets = CompiledModel_InputOffsets(compiledModel);
    for (int i = 0; i < compiledModel->inputCount; i++) {
      inputsVector[i] =
          VERSAT_OFFSET_PTR(inputs, -sutMemoryOffset + inputOffsets[i]);
    }

    printf("Inputs: %p\n", inputs);
    printf("Inputs Vector: %p\n", inputsVector);
    printf("Inputs Vector val: %p %p\n", inputsVector[0], inputsVector[1]);

    sprintf(pathBuffer, "%.*s", size, lineStart);

    FastReceiveFile(pathBuffer, "correctOutputs.bin", correct,
                    compiledModel->correctSize);
    FastReceiveFile(pathBuffer, "model.bin", model, compiledModel->modelSize);
    FastReceiveFile(pathBuffer, "inputs.bin", inputs,
                    compiledModel->totalInputSize);

    // Clear any messages from the SUT before setting start
    clear_sut_messages();
    void **sendData0 = (void **)(sutMemoryOffset + transferOffset + 0x00000000);
    void **sendData1 = (void **)(sutMemoryOffset + transferOffset + 0x00000004);
    void **sendData2 = (void **)(sutMemoryOffset + transferOffset + 0x00000008);
    void **sendData3 = (void **)(sutMemoryOffset + transferOffset + 0x0000000c);
    void **sendData4 = (void **)(sutMemoryOffset + transferOffset + 0x00000010);
    void **sendData5 = (void **)(sutMemoryOffset + transferOffset + 0x00000014);

    *sendData0 = VERSAT_OFFSET_PTR(compiledModel, -sutMemoryOffset);
    *sendData1 = VERSAT_OFFSET_PTR(output, -sutMemoryOffset);
    *sendData2 = VERSAT_OFFSET_PTR(temp, -sutMemoryOffset);
    *sendData3 = VERSAT_OFFSET_PTR(model, -sutMemoryOffset);
    *sendData4 = VERSAT_OFFSET_PTR(inputsVector, -sutMemoryOffset);
    *sendData5 = VERSAT_OFFSET_PTR(correct, -sutMemoryOffset);

    printf("Gonna clear cache\n");
    clear_cache();

    printf("Cache was cleared\n");

    clear_sut_messages();
    iob_regfileif_csrs_set_start(1);

    while (iob_regfileif_csrs_get_start() != 0) {
      relay_messages();
    }
    while (iob_regfileif_csrs_get_done() != 1) {
      relay_messages();
    }
    clear_sut_messages();

    printf("\n");
  }

  uart_puts("\n");
  uart_puts("[Tester]: Finished processing\n");

  uart_puts("[Tester]: 123\n");

  // End UART1 connection with SUT
  iob_uart_csrs_init_baseaddr(UART1_BASE);
  uart_finish();

  // Switch back to UART0
  iob_uart_csrs_init_baseaddr(UART0_BASE);

  // End UART0 connection
  uart_finish();

  return 0;
}
