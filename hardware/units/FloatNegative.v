`timescale 1ns / 1ps

module FloatNegative  #(
    parameter DATA_W = 32
    )
   (
   //control
   input clk,
   input rst,

   input running,
   input run,

   input [DATA_W-1:0] in0,

   (* versat_latency = 0 *) output [DATA_W-1:0] out0
);

assign out0 = in0[31] ? {31'b0,1'b1} : 32'b0;

endmodule