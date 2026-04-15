`timescale 1ns / 1ps

module ExtractExpAsFloat (
  input [31:0] in0,

  output [31:0] out0
  );

wire [  7:0] exponent = in0[30:23];
wire [7:0] noBiasExp = exponent - 127;

reg [7:0] outExp;
reg [22:0] outMantissa;

wire negativeOutput = noBiasExp[7];
wire [7:0] trueBias = negativeOutput ? -noBiasExp : noBiasExp;

wire outSign = negativeOutput;

always @* begin
  outExp = 0;
  outMantissa = 0;

  if(trueBias == 1) begin
    outExp = 127;
    outMantissa = 0;
  end
  if(trueBias[1] == 1'b1) begin
    outExp = 128;
    outMantissa[22] = trueBias[0];
  end
  if(trueBias[2] == 1'b1) begin
    outExp = 129;
    outMantissa[22:21] = trueBias[1:0];
  end
  if(trueBias[3] == 1'b1) begin
    outExp = 130;
    outMantissa[22:20] = trueBias[2:0];
  end
  if(trueBias[4] == 1'b1) begin
    outExp = 131;
    outMantissa[22:19] = trueBias[3:0];
  end
  if(trueBias[5] == 1'b1) begin
    outExp = 132;
    outMantissa[22:18] = trueBias[4:0];
  end
  if(trueBias[6] == 1'b1) begin
    outExp = 133;
    outMantissa[22:17] = trueBias[5:0];
  end

  if(trueBias == 8'h80) begin
    outExp = 255;
  end
end

assign out0 = {outSign,outExp,outMantissa};

endmodule
