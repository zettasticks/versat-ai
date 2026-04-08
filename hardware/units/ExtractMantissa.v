`timescale 1ns / 1ps

module ExtractMantissa #(
    parameter BITS_TO_PRESERVE = 12
  )
 (
  input [31:0] in0,

  output reg [31:0] out0
  );

wire         sign     = in0[31];
wire [  7:0] exponent = in0[30:23];
wire [ 22:0] mantissa = in0[22:0];

always @* begin
  out0 = 0;
  out0[0+:BITS_TO_PRESERVE] = mantissa[22-:BITS_TO_PRESERVE];
end

endmodule
