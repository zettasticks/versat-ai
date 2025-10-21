`timescale 1ns / 1ps

// verilator coverage_off
module iob_fp_mul_tb (

);
  localparam DATA_W = 32;
  localparam EXP_W = 8;
  // Inputs
  reg [(1)-1:0] clk_i;
  reg [(1)-1:0] rst_i;
  reg [(1)-1:0] start_i;
  reg [(DATA_W)-1:0] op_a_i;
  reg [(DATA_W)-1:0] op_b_i;
  // Outputs
  reg [(1)-1:0] done_o;
  reg [(1)-1:0] overflow_o;
  reg [(1)-1:0] underflow_o;
  reg [(1)-1:0] exception_o;
  reg [(DATA_W)-1:0] res_o;

  reg [(DATA_W)-1:0] op_a_int;
  reg [(DATA_W)-1:0] op_b_int;

  integer op_a;
  integer op_b;

  localparam CLOCK_PERIOD = 10;

  initial clk_i = 0;
  always #(CLOCK_PERIOD/2) clk_i = ~clk_i;
  `define ADVANCE @(posedge clk_i) #(CLOCK_PERIOD/2);

  iob_fp_mul #(
      .DATA_W(DATA_W),
      .EXP_W(EXP_W)
  ) uut (
    .clk_i(clk_i),
    .rst_i(rst_i),
    .start_i(start_i),
    .done_o(done_o),
    .op_a_i(op_a_i),
    .op_b_i(op_b_i),
    .overflow_o(overflow_o),
    .underflow_o(underflow_o),
    .exception_o(exception_o),
    .res_o(res_o)
  );

  task Mul (input [DATA_W-1:0] m_a, input [DATA_W-1:0] m_b);
  begin

    `ADVANCE;

    start_i = 1;
    op_a_i = m_a;
    op_b_i = m_b;

    `ADVANCE;

    start_i = 0;

    while(done_o == 0) begin
        `ADVANCE;
    end

  end
  endtask


  initial begin
    `ifdef VCD;
    $dumpfile("uut.vcd");
    $dumpvars();
    `endif // VCD;
    clk_i = 0;
    rst_i = 0;
    start_i = 0;
    op_a_i = 0;
    op_b_i = 0;

    `ADVANCE;

    rst_i = 1;

    `ADVANCE;

    rst_i = 0;

    `ADVANCE;

    // special coverage
    for(op_a=0;op_a<4;op_a=op_a+1) begin
        op_a_int = { {(EXP_W+1){op_a[1]}}, {(DATA_W-EXP_W-1){op_a[0]}} };
        Mul(op_a_int, {DATA_W{1'b0}});
    end
    op_a_i = 0;
    for(op_b=0;op_b<4;op_b=op_b+1) begin
        op_b_int = { {(EXP_W+1){op_b[1]}}, {(DATA_W-EXP_W-1){op_b[0]}} };
        Mul({DATA_W{1'b0}}, op_b_int);
    end
    op_b_i = 0;

    // for(exp=0;exp<(2**EXP_W);exp=exp+1) begin
    //     exponent_i = exp[EXP_W-1:0];
    //     for(m_b=0;m_b<(2**BOT_MANTISSA_W);m_b=m_b+1) begin
    //         mantissa_i = { {TOP_MANTISSA_W{1'b1}}, {m_b[BOT_MANTISSA_W-1:0]} };
    //         `ADVANCE;
    //         mantissa_i = { {TOP_MANTISSA_W{1'b0}}, {m_b[BOT_MANTISSA_W-1:0]} };
    //         `ADVANCE;
    //     end
    // end

    `ADVANCE;

    $finish();
  end

endmodule
// verilator coverage_on
