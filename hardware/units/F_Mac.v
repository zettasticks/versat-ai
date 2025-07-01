`timescale 1ns / 1ps

//`include "ieee754_defs.vh"

//
//     IEEE-754<DATA_W, EXP_W>
//
// |<----------DATA_W----------->|
// |_____ _________ _____________|
// |__S__|___EXP___|______F______|
// |     |         |             |
// |<-1->|<-EXP_W->|<----F_W---->|
//

`define BIAS  {1'b0, {(EXP_W-1){1'b1}}}
`define F_W   (DATA_W - EXP_W - 1'b1)
`define MAN_W (`F_W + 1'b1)

`define EXP_MAX {{(EXP_W-1){1'b1}}, 1'b0}
`define EXP_MIN {{(EXP_W-1){1'b0}}, 1'b1}
`define EXP_INF {EXP_W{1'b1}}
`define EXP_NAN {EXP_W{1'b1}}
`define EXP_SUB {EXP_W{1'b0}}

`define EXTRA 3

// Canonical NAN
`define NAN {1'b0, `EXP_NAN, 1'b1, {(DATA_W-EXP_W-2){1'b0}}}

// Infinite
`define INF(SIGN) {SIGN, `EXP_INF, {(DATA_W-EXP_W-1){1'b0}}}

module F_Mac  #(
    parameter DATA_W = 32,
    parameter EXP_W = 8,
    parameter GUARD_W = 0
    )
   (
   //control
   input clk,
   input rst,

   input running,
   input run,

   input [DATA_W-1:0] in0,
   input [DATA_W-1:0] in1,

   (* versat_latency = 30 *) output [DATA_W-1:0] out0
   );

wire sign0,sign1,signOut,invOut;
wire [EXP_W-1:0] exp0,exp1;
wire [`MAN_W-1:0] man0,man1;

wire [EXP_W+1:0] expOut;
wire [`MAN_W+`EXTRA-1:0] manOut;

ieee754_unpack unpack0(
    .clk(clk),
    .rst(rst),

    .start(1'bx),
    .done(),

    .data_i(in0),

    .sign(sign0),
    .exponent(exp0),
    .mantissa(man0)
    );

ieee754_unpack unpack1(
    .clk(clk),
    .rst(rst),

    .start(1'bx),
    .done(),

    .data_i(in1),

    .sign(sign1),
    .exponent(exp1),
    .mantissa(man1)
    );

reg t1,t2,t3,t4,t5,t6;

always @(posedge clk,posedge rst) begin
    if(rst) begin
        t1 <= 1'b0;
        t2 <= 1'b0;
        t3 <= 1'b0;
        t4 <= 1'b0;
        t5 <= 1'b0;
        t6 <= 1'b0;
    end else begin
        t1 <= t2;
        t2 <= t3;
        t3 <= t4;
        t4 <= t5;
        t5 <= t6;
        //t6 <= t1;

        if(run) begin
            t1 <= 1'b0;
            t2 <= 1'b0;
            t3 <= 1'b1;
            t4 <= 1'b1;
            t5 <= 1'b1;
            t6 <= 1'b1;
        end
    end
end

fp_mac maccer(
    .clk(clk),
    .rst(rst),

    .store(t1),
    .start(t1),
    .done(),

    .op_a_sign(sign0),
    .op_a_exp(exp0),
    .op_a_man(man0),

    .op_b_sign(sign1),
    .op_b_exp(exp1),
    .op_b_man(man1),

    .res_sign(signOut),
    .res_exp(expOut),
    .res_man(manOut),
    .res_inv(invOut)
    );

ieee754_pack packOut(
    .clk(clk),
    .rst(rst),

    .start(1'bx),
    .done(),

    .inv(invOut),

    .sign_i(signOut),
    .exp_i(expOut),
    .man_i(manOut),

    .data_o(out0)
    );

endmodule

`undef BIAS
`undef F_W
`undef MAN_W
`undef EXP_MAX
`undef EXP_MIN
`undef EXP_INF
`undef EXP_NAN
`undef EXP_SUB
`undef EXTRA
`undef NAN
`undef INF
