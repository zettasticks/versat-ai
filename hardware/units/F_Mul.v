`timescale 1ns / 1ps

module F_Mul  #(
    parameter DATA_W = 32
    )
   (
   //control
   input clk,
   input rst,

   input running,
   input run,

   input [DATA_W-1:0] in0,
   input [DATA_W-1:0] in1,

   input [6:0] delay0,

   (* versat_latency = 4 *) output [DATA_W-1:0] out0
   );

reg [6:0] delay;
always @(posedge clk, posedge rst) begin
  if (rst) begin
     delay <= 0;
  end else if (run) begin
     delay <= delay0 + 4;
  end else if (|delay) begin
     delay <= delay - 1;
  end
end

wire [DATA_W-1:0] muller_out;

iob_fp_mul muller (
  .start_i(1'b1),
  .done_o (),

  .op_a_i(in0),
  .op_b_i(in1),

  .res_o(muller_out),

  .overflow_o(),
  .underflow_o(),
  .exception_o(),

  .clk_i(clk),
  .rst_i(rst)
);

assign out0 = (running && (|delay) == 0) ? muller_out : 0;

endmodule
