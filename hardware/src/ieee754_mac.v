`timescale 1ns / 1ps


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


module ieee754_mac
  #(
    parameter DATA_W = 32,
    parameter EXP_W = 8,
    parameter GUARD_W = 0
    )
   (
    input               clk,
    input               rst,

    input               start,
    output              done,

    input [DATA_W-1:0]  op_a,
    input [DATA_W-1:0]  op_b,

    output [DATA_W-1:0] res
    );

   wire                 op_a_sign;
   wire [EXP_W-1:0]     op_a_exp;
   wire [`MAN_W-1:0]    op_a_man;
   wire                 op_a_nan;
   wire                 op_a_inf;
   wire                 op_a_zero;
   wire                 done_unpack0;
   ieee754_unpack
     #(
       .DATA_W (DATA_W),
       .EXP_W  (EXP_W)
       )
   unpack0
       (
        .clk      (clk),
        .rst      (rst),
   
        .start    (start),
        .done     (done_unpack0),

        .data_i   (op_a),

`ifdef SPECIAL_CASES
        .nan      (op_a_nan),
        .infinite (op_a_inf),
        .zero     (op_a_zero),
`endif
   
        .sign     (op_a_sign),
        .exponent (op_a_exp),
        .mantissa (op_a_man)
        );

   wire                 op_b_sign;
   wire [EXP_W-1:0]     op_b_exp;
   wire [`MAN_W-1:0]    op_b_man;
   wire                 op_b_nan;
   wire                 op_b_inf;
   wire                 op_b_zero;
   wire                 done_unpack1;
   ieee754_unpack
     #(
       .DATA_W (DATA_W),
       .EXP_W  (EXP_W)
       )
   unpack1
       (
        .clk      (clk),
        .rst      (rst),
   
        .start    (start),
        .done     (done_unpack1),

        .data_i   (op_b),

`ifdef SPECIAL_CASES
        .nan      (op_b_nan),
        .infinite (op_b_inf),
        .zero     (op_b_zero),
`endif
   
        .sign     (op_b_sign),
        .exponent (op_b_exp),
        .mantissa (op_b_man)
        );

   wire                 start_mac = done_unpack0;
   wire                 done_mac;
   wire                 mac_sign;
   wire [EXP_W+1:0]     mac_exp;
   wire [`MAN_W+`EXTRA-1:0] mac_man;
   wire                     mac_inv;
   fp_mac
     #(
       .DATA_W  (DATA_W),
       .EXP_W   (EXP_W),
       .GUARD_W (GUARD_W)
       )
   mac0
       (
        .clk         (clk),
        .rst         (rst),

        .start       (start_mac),
        .done        (done_mac),

        .op_a_sign   (op_a_sign),
        .op_a_exp    (op_a_exp),
        .op_a_man    (op_a_man),

        .op_b_sign   (op_b_sign),
        .op_b_exp    (op_b_exp),
        .op_b_man    (op_b_man),

        .res_sign    (mac_sign),
        .res_exp     (mac_exp),
        .res_man     (mac_man),
        .res_inv     (mac_inv)
        );

   wire                     start_pack = done_mac;
   wire [DATA_W-1:0]        res_int;
   ieee754_pack
     #(
       .DATA_W (DATA_W),
       .EXP_W  (EXP_W)
       )
   pack0
       (
        .clk      (clk),
        .rst      (rst),
   
        .start    (start_pack),
        .done     (done),

        .inv      (mac_inv),

        .sign_i   (mac_sign),
        .exp_i    (mac_exp),
        .man_i    (mac_man),

        .data_o   (res)
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

// Canonical NAN
`undef NAN

// Infinite
`undef INF

`undef SPECIAL_CASES


