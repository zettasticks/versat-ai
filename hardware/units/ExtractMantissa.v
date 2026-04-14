`timescale 1ns / 1ps

module ExtractMantissa #(
    parameter BITS_TO_PRESERVE = 12
  )
 (
  input [31:0] in0,

  output reg [31:0] out0
  );

wire [BITS_TO_PRESERVE-1:0] mantissa = in0[BITS_TO_PRESERVE-1:0];

always @* begin
  out0 = 0;
  out0[0+:BITS_TO_PRESERVE] = mantissa;
end

endmodule
