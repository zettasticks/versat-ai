`timescale 1ns / 1ps

/* verilator lint_off WIDTH */

module MyFloatAccum #(
   parameter DATA_W = 32,
   parameter STRIDE_W = 16,
   parameter DELAY_W = 7
) (
   //control
   input clk,
   input rst,

   input run,
   input running,

   input [STRIDE_W-1:0] strideMinusOne,

   input [DATA_W-1:0] in0,

   (* versat_latency = 3 *) output reg [31:0] out0,

   input [DELAY_W-1:0] delay0
);

   reg [31:0] delay;

   always @(posedge clk, posedge rst) begin
      if (rst) begin
         delay <= 0;
      end else if (run) begin
         delay <= {25'h0,delay0};
      end else if (|delay) begin
         delay <= delay - 1;
      end else begin
         delay <= {16'b0,strideMinusOne};
      end
   end

   wire doAccum = (delay == 0);

   reg [278:0] res;

   localparam MAN_W = 23;
   localparam EXP_W = 8;
   localparam BIAS = 2 ** (EXP_W - 1) - 1;

   wire [278:0] in0_decoded;
   reg  [278:0] accum;
   reg  doAccum_2;
   reg [31:0] valueUsedForAccumReset;

   // Stage 1 - Decode in0
   FloatToLargeInteger conv (
      .in_i (in0),
      .out_o(in0_decoded)
   );

   reg [278:0] in0_decoded_reg;
   always @(posedge clk, posedge rst) begin
      if (rst) begin
         doAccum_2 <= 0;
         in0_decoded_reg <= 0;
      end else if (running) begin
         doAccum_2 <= doAccum;
         in0_decoded_reg <= in0_decoded;
      end
   end

   // Stage 2 - Calculate accum
   always @(posedge clk, posedge rst) begin
      if (rst) begin
         accum <= 0;
         valueUsedForAccumReset <= 0;
      end else begin
         if (doAccum_2) begin
            accum <= in0_decoded_reg;
            valueUsedForAccumReset <= in0_decoded_reg; 
         end else if (running) begin
            accum <= in0_decoded_reg + accum;
         end
      end
   end

   // Stage 3 - Calculate exponent for accum

   // Repack
   wire [8:0] lzc, nlzc;
   wire [278:0] negatedAccum = -accum;
   clz #(
      .DATA_W(279)
   ) count (
      .data_i(accum),
      .data_o(lzc)
   );
   clz #(
      .DATA_W(279)
   ) countNeg (
      .data_i(negatedAccum),
      .data_o(nlzc)
   );

   reg                             sign_reg;
   reg                     [  7:0] exponent;
   reg                     [278:0] accum_reg;
   reg                     [278:0] accumNeg_reg;

   wire sign = accum[278];

   always @(posedge clk, posedge rst) begin
      if (rst) begin
         sign_reg     <= 0;
         accum_reg    <= 0;
         accumNeg_reg <= 0;
         exponent     <= 0;
      end else if (running) begin
         accum_reg    <= accum;
         accumNeg_reg <= negatedAccum;
         sign_reg     <= sign;

         if (lzc == 9'd279) begin
            exponent <= 0;
         end else begin
            if (sign) begin
               exponent <= 8'hff - nlzc[7:0];
            end else begin
               exponent <= 8'hff - lzc[7:0];
            end
         end
      end
   end

   // Stage 4 - Calculate final values

   reg        signal_final;
   reg [ 7:0] exponent_final;
   reg [22:0] mantissa_final;

   always @(posedge clk, posedge rst) begin
      if (rst) begin
         signal_final   <= 0;
         exponent_final <= 0;
         mantissa_final <= 0;
      end else if (running) begin
         signal_final   <= sign_reg;
         exponent_final <= exponent;

         if (sign_reg) begin
            mantissa_final <= accumNeg_reg[{1'b0, exponent}+:23];
         end else begin
            mantissa_final <= accum_reg[{1'b0, exponent}+:23];
         end
      end
   end

   // Output

   assign out0 = {signal_final, exponent_final, mantissa_final};

endmodule