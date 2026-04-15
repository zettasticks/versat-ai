`timescale 1ns / 1ps

module F_IsNeg (
  input [31:0] in0,

  output [31:0] out0
  );

assign out0 = in0[31] ? 32'hffffffff : 32'h00000000;

endmodule
