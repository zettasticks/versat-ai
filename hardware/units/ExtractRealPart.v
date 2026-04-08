`timescale 1ns / 1ps

module ExtractRealPart (
  input [31:0] in0,

  output [31:0] out0, // Real part as an integer + sign bit on the 7 index
  output [31:0] out1  // Floating point format of the real part (If in0 is 12.45, out1 is floating point 12.0)
  );

wire         sign     = in0[31];
wire [  7:0] exponent = in0[30:23];
wire [ 22:0] mantissa = in0[22:0];

wire [31:0] noSign = {1'b0,exponent,mantissa};
wire [15:0] half = noSign[31:16];

reg [6:0] realPart;
reg [5:0] mantissaPart;

always @* begin
  realPart = 0;
  mantissaPart = 0;

  if(exponent == 8'h7f) begin
    realPart = 1;
  end
  if(exponent == 8'h80) begin
    realPart = 2 + {6'b0,mantissa[22]};
    mantissaPart[5] = mantissa[22];
  end
  if(exponent == 8'h81) begin
    realPart = 4 + {5'b0,mantissa[22:21]};
    mantissaPart[5:4] = mantissa[22:21];
  end
  if(exponent == 8'h82) begin
    realPart = 8 + {4'b0,mantissa[22:20]};
    mantissaPart[5:3] = mantissa[22:20];
  end
  if(exponent == 8'h83) begin
    realPart = 16 + {3'b0,mantissa[22:19]};
    mantissaPart[5:2] = mantissa[22:19];
  end
  if(exponent == 8'h84) begin
    realPart = 32 + {2'b0,mantissa[22:18]};
    mantissaPart[5:1] = mantissa[22:18];
  end
  if(exponent == 8'h85) begin
    realPart = 64 + {1'b0,mantissa[22:17]};
    mantissaPart = mantissa[22:17];
  end
end

assign out0 = {24'h0,sign,realPart};
assign out1 = (realPart == 0) ? 32'h0 : {sign,exponent,{mantissaPart,17'b0}};

endmodule
