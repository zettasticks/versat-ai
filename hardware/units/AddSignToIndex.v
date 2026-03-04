`timescale 1ns / 1ps

module AddSignToIndex #(
    parameter INDEX_BIT = 0
  ) (
  input [31:0] in0,
  input [31:0] in1, // Negative

  output [31:0] out0
  );

assign out0 = (|in1) ? (in0 + (1 << INDEX_BIT)) : in0;

endmodule
