/*
 * SPDX-FileCopyrightText: 2025 IObundle
 *
 * SPDX-License-Identifier: MIT
 */

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
#else                      // NOT IOB_SYSTEM_TESTER_USE_ETHERNET
  // Receive data from console via UART
  file_size =
      uart_recvfile("../../../software/versat_ai_firmware.bin",
                    (char *)0x40000000); // Place SUT fw in external memory
#endif                     // IOB_SYSTEM_TESTER_USE_ETHERNET

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
  // This alternative is better to see satus of SUT in real time, but tester may miss/skip some characters if it cant read them fast enough. One solution to avoid skipping characters is using a UART that includes a FIFO (like uart16550).
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
