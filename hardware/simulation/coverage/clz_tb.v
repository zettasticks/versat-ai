`timescale 1ns / 1ps

// verilator coverage_off
module clz_tb (

);
  localparam DATA_W = 16;
  localparam OUT_W = 4;
  // Inputs
  reg [(DATA_W)-1:0] data_i;
  // Outputs
  reg [(OUT_W)-1:0] data_o;

  integer i;

  `define ADVANCE #(10);
  clz #(
    .DATA_W(DATA_W),
    .OUT_W(OUT_W)
  ) uut (
    .data_i(data_i),
    .data_o(data_o)
  );


  initial begin
    `ifdef VCD;
    $dumpfile("uut.vcd");
    $dumpvars();
    `endif // VCD;
    data_i = 0;

    for(i=0;i<(2**DATA_W);i=i+1) begin
      data_i = i[DATA_W-1:0];
      `ADVANCE;
    end

    `ADVANCE;

    data_i = 0;

    `ADVANCE;

    $finish();
  end

endmodule
// verilator coverage_on
