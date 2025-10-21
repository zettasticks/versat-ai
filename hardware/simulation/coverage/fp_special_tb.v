`timescale 1ns / 1ps

// verilator coverage_off
module fp_special_tb (

);
  localparam DATA_W = 32;
  localparam EXP_W = 8;
  // Inputs
  reg [(DATA_W)-1:0] data_i;
  // Outputs
  reg [(1)-1:0] nan;
  reg [(1)-1:0] infinite;
  reg [(1)-1:0] zero;

  integer i;

  `define ADVANCE #(10);
  fp_special #(
      .DATA_W(DATA_W),
      .EXP_W(EXP_W)
  ) uut (
    .data_i(data_i),
    .nan(nan),
    .infinite(infinite),
    .zero(zero)
  );


  initial begin
    `ifdef VCD;
    $dumpfile("uut.vcd");
    $dumpvars();
    `endif // VCD;
    data_i = 0;

    for(i=0;i<4;i=i+1) begin
        data_i = { {(EXP_W+1){i[1]}}, {(DATA_W-EXP_W-1){i[0]}} };
        `ADVANCE;
    end

    data_i = 0;

    `ADVANCE;

    $finish();
  end

endmodule
// verilator coverage_on
