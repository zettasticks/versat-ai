`timescale 1ns / 1ps

// verilator coverage_off
module FloatToLargeInteger_tb (

);
  localparam IN_W = 32;
  localparam OUT_W = 279;
  // Inputs
  reg [(IN_W)-1:0] in_i;
  // Outputs
  reg [(OUT_W)-1:0] out_o;

  integer i;

  `define ADVANCE #(10);
  FloatToLargeInteger 
  uut (
    .in_i(in_i),
    .out_o(out_o)
  );


  initial begin
    `ifdef VCD;
    $dumpfile("uut.vcd");
    $dumpvars();
    `endif // VCD;
    in_i = 0;

    for(i=0;i<32;i=i+1) begin

        in_i = (1<<i);
        `ADVANCE;
        in_i = 0;
        `ADVANCE;
    end

    `ADVANCE;

    in_i = 0;

    `ADVANCE;

    $finish();
  end

endmodule
// verilator coverage_on
