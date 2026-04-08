`timescale 1ns / 1ps

module FracIndex #(
    parameter PRECISION = 12
  ) (
  input [31:0] in0,

  output [31:0] out0
);

// u32 fracIndex = (unpacked.mantissa >> (8 + (16 - fractionalPrecision) - unpacked.exponent)) & ((1 << fractionalPrecision) - 1);

wire         sign     = in0[31];
wire [  7:0] exponent = in0[30:23];
wire [ 22:0] mantissa = in0[22:0];

wire [ 7:0]  shiftAmount = (8 + (16 - PRECISION) - (exponent - 126));
wire [31:0]  shifted = ({8'b0,1'b1,mantissa} >> shiftAmount);

assign out0 = { {32-PRECISION{1'b0}} , shifted[PRECISION-1:0] };

endmodule