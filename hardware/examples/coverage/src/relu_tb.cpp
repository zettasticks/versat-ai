#include <cstdio>
#include <verilated.h>
#if (VM_TRACE == 1) // If verilator was invoked with --trace
#if (VM_TRACE_FST == 1)
#include <verilated_fst_c.h>
#else
#include <verilated_vcd_c.h>
#endif
#endif

#include "VRelu.h" //user file that defins the dut

#ifndef CLK_PERIOD
#define FREQ 100000000
#define CLK_PERIOD 1000000000 / FREQ // Example: 1/100MHz*10^9 = 10 ns
#endif

#if (VM_TRACE == 1)
#if (VM_TRACE_FST == 1)
VerilatedFstC *tfp = new VerilatedFstC; // Create tracing object
#else
VerilatedVcdC *tfp = new VerilatedVcdC; // Create tracing object
#endif
#endif

// simulation time
vluint64_t sim_time = 0;

// Delayed start time of VCD trace dump
// Used to avoid large VCD dump files during long simulations
#if (VM_TRACE == 1)
vluint64_t vcd_delayed_start = 0;
#endif

VRelu *dut = new VRelu; // Create instance of module

// Clock tick
void clk_tick(unsigned int n = 1) {
  for (unsigned int i = 0; i < n; i++) {
    dut->eval();
#if (VM_TRACE == 1)
    tfp->dump(sim_time); // Dump values into tracing file
#endif
    sim_time += CLK_PERIOD / 2;
    dut->clk = !dut->clk; // negedge
    dut->eval();
#if (VM_TRACE == 1)
    tfp->dump(sim_time);
#endif
    sim_time += CLK_PERIOD / 2;
    dut->clk = !dut->clk; // posedge
    dut->eval();
  }
}

// Reset dut
void iob_hard_reset() {
  dut->clk = 1;
  dut->rst = 0;
  clk_tick(100);
  dut->rst = 1;
  clk_tick(100);
  dut->rst = 0;
  clk_tick(100);
}

// write all possible values into input width
void cycle_inputs(unsigned int input_w) {
  unsigned long max = ((1UL) << input_w);
  unsigned int loop = 0, bit = 0, in0 = 0;
  dut->run = 0;
  dut->running = 0;
  std::printf("INFO: Cycling inputs for input width %u\n", input_w);
  for (loop = 0; loop < 2; loop++) {
    for (bit = 0; bit < input_w; bit++) {
      unsigned int mask = (1 << bit);
      in0 = (in0 ^ mask); // toggle in0[bit]
      printf("\tINFO: in0: 0x%08X\n", in0);
      dut->run = 1;
      dut->running = 1;
      dut->in0 = in0;
      clk_tick();
    }
  }
  dut->in0 = 0;
  clk_tick();
  dut->run = 0;
  dut->running = 0;
  clk_tick(5);
}

int main(int argc, char **argv) {

  Verilated::commandArgs(argc, argv); // Init verilator context

#if (VM_TRACE == 1)
  Verilated::traceEverOn(true); // Enable tracing
  dut->trace(tfp, 1);
  tfp->open("uut.vcd");
#endif

  // hardware reset
  iob_hard_reset();

  // cycle over all input
  // cycle_inputs(10); // set lower coverage
  cycle_inputs(32); // toggle all bits

  // terminate simulation and generate trace file
  dut->final();

#if (VM_TRACE == 1)
  tfp->close(); // Close tracing file
  fprintf(stdout, "Trace file created: uut.vcd\n");
  delete tfp;
#endif

  // Coverage analysis (calling write only after the test is known to pass)
#if VM_COVERAGE
  Verilated::mkdir("logs");
  Verilated::threadContextp()->coveragep()->write("logs/coverage.dat");
#endif

  delete dut;

  return 0;
}
