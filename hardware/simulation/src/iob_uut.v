
// SPDX-FileCopyrightText: 2025 IObundle, Lda
//
// SPDX-License-Identifier: MIT
//
// Py2HWSW Version 0.81 has generated this code (https://github.com/IObundle/py2hwsw).

`timescale 1ns / 1ps
`include "iob_uut_conf.vh"

module iob_uut #(
    parameter AXI_ID_W = `IOB_UUT_AXI_ID_W,  // Don't change this parameter value!
    parameter AXI_LEN_W = `IOB_UUT_AXI_LEN_W,  // Don't change this parameter value!
    parameter AXI_ADDR_W = `IOB_UUT_AXI_ADDR_W,  // Don't change this parameter value!
    parameter AXI_DATA_W = `IOB_UUT_AXI_DATA_W,  // Don't change this parameter value!
    parameter BAUD = `IOB_UUT_BAUD,  // Don't change this parameter value!
    parameter FREQ = `IOB_UUT_FREQ,  // Don't change this parameter value!
    parameter SIMULATION = `IOB_UUT_SIMULATION  // Don't change this parameter value!
) (
    // clk_en_rst_s: Clock, clock enable and reset
    input clk_i,
    input cke_i,
    input arst_i,
    // uart_s: Testbench uart csrs interface
    input iob_valid_i,
    input [3-1:0] iob_addr_i,
    input [32-1:0] iob_wdata_i,
    input [32/8-1:0] iob_wstrb_i,
    output iob_rvalid_o,
    output [32-1:0] iob_rdata_o,
    output iob_ready_o
);

// rs232 bus
    wire rs232_rxd;
    wire rs232_txd;
    wire rs232_rts;
    wire rs232_cts;
// AXI bus to connect SoC to interconnect
    wire [AXI_ADDR_W-1:0] axi_araddr;
    wire axi_arvalid;
    wire axi_arready;
    wire [AXI_DATA_W-1:0] axi_rdata;
    wire [2-1:0] axi_rresp;
    wire axi_rvalid;
    wire axi_rready;
    wire [AXI_ID_W-1:0] axi_arid;
    wire [AXI_LEN_W-1:0] axi_arlen;
    wire [3-1:0] axi_arsize;
    wire [2-1:0] axi_arburst;
    wire axi_arlock;
    wire [4-1:0] axi_arcache;
    wire [4-1:0] axi_arqos;
    wire [AXI_ID_W-1:0] axi_rid;
    wire axi_rlast;
    wire [AXI_ADDR_W-1:0] axi_awaddr;
    wire axi_awvalid;
    wire axi_awready;
    wire [AXI_DATA_W-1:0] axi_wdata;
    wire [AXI_DATA_W/8-1:0] axi_wstrb;
    wire axi_wvalid;
    wire axi_wready;
    wire [2-1:0] axi_bresp;
    wire axi_bvalid;
    wire axi_bready;
    wire [AXI_ID_W-1:0] axi_awid;
    wire [AXI_LEN_W-1:0] axi_awlen;
    wire [3-1:0] axi_awsize;
    wire [2-1:0] axi_awburst;
    wire axi_awlock;
    wire [4-1:0] axi_awcache;
    wire [4-1:0] axi_awqos;
    wire axi_wlast;
    wire [AXI_ID_W-1:0] axi_bid;
// Connect axi_ram to 'iob_ram_t2p_be' memory
    wire ext_mem_clk;
    wire ext_mem_r_en;
    wire [AXI_ADDR_W - 2-1:0] ext_mem_r_addr;
    wire [32-1:0] ext_mem_r_data;
    wire [32/8-1:0] ext_mem_w_strb;
    wire [AXI_ADDR_W - 2-1:0] ext_mem_w_addr;
    wire [32-1:0] ext_mem_w_data;

        // IOb-SoC memory wrapper
        versat_ai_mwrap #(
        .AXI_ID_W(AXI_ID_W),
        .AXI_LEN_W(AXI_LEN_W),
        .AXI_ADDR_W(AXI_ADDR_W),
        .AXI_DATA_W(AXI_DATA_W)
    ) iob_memwrapper (
            // clk_en_rst_s port: Clock, clock enable and reset
        .clk_i(clk_i),
        .cke_i(cke_i),
        .arst_i(arst_i),
        // rs232_m port: iob-system uart interface
        .rs232_rxd_i(rs232_rxd),
        .rs232_txd_o(rs232_txd),
        .rs232_rts_o(rs232_rts),
        .rs232_cts_i(rs232_cts),
        // axi_m port: AXI manager interface for DDR memory
        .axi_araddr_o(axi_araddr),
        .axi_arvalid_o(axi_arvalid),
        .axi_arready_i(axi_arready),
        .axi_rdata_i(axi_rdata),
        .axi_rresp_i(axi_rresp),
        .axi_rvalid_i(axi_rvalid),
        .axi_rready_o(axi_rready),
        .axi_arid_o(axi_arid),
        .axi_arlen_o(axi_arlen),
        .axi_arsize_o(axi_arsize),
        .axi_arburst_o(axi_arburst),
        .axi_arlock_o(axi_arlock),
        .axi_arcache_o(axi_arcache),
        .axi_arqos_o(axi_arqos),
        .axi_rid_i(axi_rid),
        .axi_rlast_i(axi_rlast),
        .axi_awaddr_o(axi_awaddr),
        .axi_awvalid_o(axi_awvalid),
        .axi_awready_i(axi_awready),
        .axi_wdata_o(axi_wdata),
        .axi_wstrb_o(axi_wstrb),
        .axi_wvalid_o(axi_wvalid),
        .axi_wready_i(axi_wready),
        .axi_bresp_i(axi_bresp),
        .axi_bvalid_i(axi_bvalid),
        .axi_bready_o(axi_bready),
        .axi_awid_o(axi_awid),
        .axi_awlen_o(axi_awlen),
        .axi_awsize_o(axi_awsize),
        .axi_awburst_o(axi_awburst),
        .axi_awlock_o(axi_awlock),
        .axi_awcache_o(axi_awcache),
        .axi_awqos_o(axi_awqos),
        .axi_wlast_o(axi_wlast),
        .axi_bid_i(axi_bid)
        );

            // Testbench uart core
        iob_uart uart_tb (
            // clk_en_rst_s port: Clock, clock enable and reset
        .clk_i(clk_i),
        .cke_i(cke_i),
        .arst_i(arst_i),
        // iob_csrs_cbus_s port: Control and Status Registers interface (auto-generated)
        .iob_csrs_iob_valid_i(iob_valid_i),
        .iob_csrs_iob_addr_i(iob_addr_i),
        .iob_csrs_iob_wdata_i(iob_wdata_i),
        .iob_csrs_iob_wstrb_i(iob_wstrb_i),
        .iob_csrs_iob_rvalid_o(iob_rvalid_o),
        .iob_csrs_iob_rdata_o(iob_rdata_o),
        .iob_csrs_iob_ready_o(iob_ready_o),
        // rs232_m port: RS232 interface
        .rs232_rxd_i(rs232_txd),
        .rs232_txd_o(rs232_rxd),
        .rs232_rts_o(rs232_cts),
        .rs232_cts_i(rs232_rts)
        );

            // External memory
        iob_axi_ram #(
        .ID_WIDTH(AXI_ID_W),
        .ADDR_WIDTH(AXI_ADDR_W),
        .DATA_WIDTH(AXI_DATA_W)
    ) ddr_model_mem (
            // clk_i port: Clock
        .clk_i(clk_i),
        // rst_i port: Synchronous reset
        .rst_i(arst_i),
        // axi_s port: AXI interface
        .axi_araddr_i(axi_araddr),
        .axi_arvalid_i(axi_arvalid),
        .axi_arready_o(axi_arready),
        .axi_rdata_o(axi_rdata),
        .axi_rresp_o(axi_rresp),
        .axi_rvalid_o(axi_rvalid),
        .axi_rready_i(axi_rready),
        .axi_arid_i(axi_arid),
        .axi_arlen_i(axi_arlen),
        .axi_arsize_i(axi_arsize),
        .axi_arburst_i(axi_arburst),
        .axi_arlock_i({1'b0, axi_arlock}),
        .axi_arcache_i(axi_arcache),
        .axi_arqos_i(axi_arqos),
        .axi_rid_o(axi_rid),
        .axi_rlast_o(axi_rlast),
        .axi_awaddr_i(axi_awaddr),
        .axi_awvalid_i(axi_awvalid),
        .axi_awready_o(axi_awready),
        .axi_wdata_i(axi_wdata),
        .axi_wstrb_i(axi_wstrb),
        .axi_wvalid_i(axi_wvalid),
        .axi_wready_o(axi_wready),
        .axi_bresp_o(axi_bresp),
        .axi_bvalid_o(axi_bvalid),
        .axi_bready_i(axi_bready),
        .axi_awid_i(axi_awid),
        .axi_awlen_i(axi_awlen),
        .axi_awsize_i(axi_awsize),
        .axi_awburst_i(axi_awburst),
        .axi_awlock_i({1'b0, axi_awlock}),
        .axi_awcache_i(axi_awcache),
        .axi_awqos_i(axi_awqos),
        .axi_wlast_i(axi_wlast),
        .axi_bid_o(axi_bid),
        // external_mem_bus_m port: Port for connection to external 'iob_ram_t2p_be' memory
        .ext_mem_clk_o(ext_mem_clk),
        .ext_mem_r_en_o(ext_mem_r_en),
        .ext_mem_r_addr_o(ext_mem_r_addr),
        .ext_mem_r_data_i(ext_mem_r_data),
        .ext_mem_w_strb_o(ext_mem_w_strb),
        .ext_mem_w_addr_o(ext_mem_w_addr),
        .ext_mem_w_data_o(ext_mem_w_data)
        );

            // Default description
        iob_ram_t2p_be #(
        .ADDR_W(AXI_ADDR_W - 2),
        .DATA_W(AXI_DATA_W),
        .HEXFILE("versat_ai_firmware")
    ) iob_ram_t2p_be_inst (
            // ram_t2p_be_s port: RAM interface
        .clk_i(ext_mem_clk),
        .r_en_i(ext_mem_r_en),
        .r_addr_i(ext_mem_r_addr),
        .r_data_o(ext_mem_r_data),
        .w_strb_i(ext_mem_w_strb),
        .w_addr_i(ext_mem_w_addr),
        .w_data_i(ext_mem_w_data)
        );

    
endmodule
