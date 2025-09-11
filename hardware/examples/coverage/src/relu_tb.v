`timescale 1ns/1ps

module relu_tb;
  reg         clk     = 0;
  reg         rst     = 1;
  reg         running = 0;
  reg  [31:0] in0     = 0;
  wire [31:0] out0;

  // DUT
  Relu dut (
    .clk(clk),
    .rst(rst),
    .running(running),
    .in0(in0),
    .out0(out0)
  );

  // 100 MHz clock => 10 ns period
  always #5 clk <= ~clk;

  initial begin
`ifdef VCD
      $dumpfile("uut.vcd");
      $dumpvars();
`endif

    // Hold reset for a couple of cycles
    rst = 0;
    repeat (2) @(posedge clk);
    rst = 1;
    repeat (2) @(posedge clk);
    rst = 0;
    assert(out0 == 0) else $fatal("Initial output should be 0 after reset");

    // Enable running and test negative input -> expect 0
    running = 1;

    in0 = 32'h0000_0000;  // 0
    @(posedge clk); #1;   // sample after posedge update
    assert(out0 == in0) else $fatal("Output should be in0");

    in0 = 32'h8000_0001;  // negative value
    @(posedge clk); #1;
    assert(out0 == 0) else $fatal("Output should be 0");

    in0 = 32'h0000_0005;  // Positive input -> expect same value
    @(posedge clk); #1;
    assert(out0 == in0) else $fatal("Output should be in0");

    // running=0 should hold previous output
    running = 0;
    in0 = 32'h0000_0000;  // 0
    @(posedge clk); #1;
    assert(out0 == 0) else $fatal("Output should be same as previously");

    // Re-enable and test another negative
    running = 1;
    in0 = 32'h7FFF_FFFF;  // max positive value
    @(posedge clk); #1;
    assert(out0 == in0) else $fatal("Output should be in0");

    in0 = 32'h0000_0000;  // 0
    @(posedge clk); #1;
    assert(out0 == in0) else $fatal("Output should be in0");

    in0 = 32'h7FFF_FFFF;  // max positive value
    @(posedge clk); #1;
    assert(out0 == in0) else $fatal("Output should be in0");

    $display("All tests passed.");
    $finish;
  end
endmodule
