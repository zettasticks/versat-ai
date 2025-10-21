`timescale 1ns / 1ps

// verilator coverage_off
module iob_fp_clz_tb (

);
  localparam DATA_W = 8;
  // Inputs
  reg [(DATA_W)-1:0] data_i;
  // Outputs
  reg [(DATA_W)-1:0] data_o;

  integer i;

  `define ADVANCE #(10);
  iob_fp_clz #(
    .DATA_W(DATA_W)
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
