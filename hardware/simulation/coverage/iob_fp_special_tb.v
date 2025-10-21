`timescale 1ns / 1ps

// verilator coverage_off
module iob_fp_special_tb (

);
  localparam DATA_W = 32;
  localparam EXP_W = 8;
  // Inputs
  reg [(DATA_W)-1:0] data_i;
  // Outputs
  reg [(1)-1:0] nan_o;
  reg [(1)-1:0] infinite_o;
  reg [(1)-1:0] zero_o;
  reg [(1)-1:0] sub_normal_o;

  integer i;

  `define ADVANCE #(10);
  iob_fp_special #(
      .DATA_W(DATA_W),
      .EXP_W(EXP_W)
  ) uut (
    .data_i(data_i),
    .nan_o(nan_o),
    .infinite_o(infinite_o),
    .zero_o(zero_o),
    .sub_normal_o(sub_normal_o)
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
