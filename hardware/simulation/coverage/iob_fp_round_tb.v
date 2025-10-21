`timescale 1ns / 1ps

// verilator coverage_off
module iob_fp_round_tb (

);
  localparam DATA_W = 24;
  localparam EXP_W = 8;
  // Inputs
  reg [(EXP_W)-1:0] exponent_i;
  reg [(DATA_W+3)-1:0] mantissa_i;
  // Outputs
  reg [(EXP_W)-1:0] exponent_rnd_o;
  reg [(DATA_W-1)-1:0] mantissa_rnd_o;

  localparam BOT_MANTISSA_W = 4;
  localparam TOP_MANTISSA_W = (DATA_W+3-BOT_MANTISSA_W);
  integer m_b;
  integer exp;

  `define ADVANCE #(10);
  iob_fp_round #(
      .DATA_W(DATA_W),
      .EXP_W(EXP_W)
  ) uut (
    .exponent_i(exponent_i),
    .mantissa_i(mantissa_i),
    .exponent_rnd_o(exponent_rnd_o),
    .mantissa_rnd_o(mantissa_rnd_o)
  );


  initial begin
    `ifdef VCD;
    $dumpfile("uut.vcd");
    $dumpvars();
    `endif // VCD;
    exponent_i = 0;
    mantissa_i = 0;

    for(exp=0;exp<(2**EXP_W);exp=exp+1) begin
        exponent_i = exp[EXP_W-1:0];
        for(m_b=0;m_b<(2**BOT_MANTISSA_W);m_b=m_b+1) begin
            mantissa_i = { {TOP_MANTISSA_W{1'b1}}, {m_b[BOT_MANTISSA_W-1:0]} };
            `ADVANCE;
            mantissa_i = { {TOP_MANTISSA_W{1'b0}}, {m_b[BOT_MANTISSA_W-1:0]} };
            `ADVANCE;
        end
    end

    exponent_i = 0;
    mantissa_i = 0;

    `ADVANCE;

    $finish();
  end

endmodule
// verilator coverage_on
