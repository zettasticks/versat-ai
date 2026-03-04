`timescale 1ns / 1ps

module F_Neg (
  input [31:0] in0,

  output [31:0] out0
  );

assign out0 = {!in0[31],in0[30:0]};

endmodule
