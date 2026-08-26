// SPDX-FileCopyrightText: 2026 IObundle, Lda
//
// SPDX-License-Identifier: MIT
//
// Py2HWSW Version 0.81.0 has generated this code (https://github.com/IObundle/py2hwsw).

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
    // tb_s: Testbench iob interface
    input iob_valid_i,
    input [32-1:0] iob_addr_i,
    input [32-1:0] iob_wdata_i,
    input [32/8-1:0] iob_wstrb_i,
    output iob_rvalid_o,
    output [32-1:0] iob_rdata_o,
    output iob_ready_o
);


    localparam TRUE_AXI_ADDR = 30;
    localparam OFFSET = $clog2(AXI_DATA_W/8);
    localparam MEM_DATA_W = AXI_DATA_W;


// rs232 bus
    wire rs232_rxd;
    wire rs232_txd;
    wire rs232_rts;
    wire rs232_cts;
// AXI bus to connect SoC to interconnect
    wire [AXI_ADDR_W-1:0] axi_araddr;
    wire axi_arvalid;
    wire axi_arready;
    wire [MEM_DATA_W-1:0] axi_rdata;
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
    wire [MEM_DATA_W-1:0] axi_wdata;
    wire [MEM_DATA_W/8-1:0] axi_wstrb;
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
    wire [MEM_DATA_W-1:0] ext_mem_r_data;
    wire [MEM_DATA_W/8-1:0] ext_mem_w_strb;
    wire [AXI_ADDR_W - 2-1:0] ext_mem_w_addr;
    wire [MEM_DATA_W-1:0] ext_mem_w_data;

        // IOb-SoC memory wrapper
        versat_ai_mwrap #(
        .AXI_ID_W(AXI_ID_W),
        .AXI_LEN_W(AXI_LEN_W),
        .AXI_ADDR_W(AXI_ADDR_W),
        .AXI_DATA_W(AXI_DATA_W),
        .SIMULATION(SIMULATION)
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
        // csrs_cbus_s port: Control and Status Registers interface (auto-generated)
        .csrs_iob_valid_i(iob_valid_i),
        .csrs_iob_addr_i(iob_addr_i[3:0]),
        .csrs_iob_wdata_i(iob_wdata_i),
        .csrs_iob_wstrb_i(iob_wstrb_i),
        .csrs_iob_rvalid_o(iob_rvalid_o),
        .csrs_iob_rdata_o(iob_rdata_o),
        .csrs_iob_ready_o(iob_ready_o),
        // rs232_m port: RS232 interface
        .rs232_rxd_i(rs232_txd),
        .rs232_txd_o(rs232_rxd),
        .rs232_rts_o(rs232_cts),
        .rs232_cts_i(rs232_rts)
        );

            // External memory
        iob_axi_ram #(
        .ID_WIDTH(AXI_ID_W),
        .ADDR_WIDTH(TRUE_AXI_ADDR),
        .DATA_WIDTH(MEM_DATA_W)
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
        .ADDR_W(TRUE_AXI_ADDR - OFFSET),
        .DATA_W(MEM_DATA_W),
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

//axi_adapter_direct #
//(
//    .ADDR_WIDTH(TRUE_AXI_ADDR),
//    .S_DATA_WIDTH(32),
//    .M_DATA_WIDTH(MEM_DATA_W),
//    .CONVERT_NARROW_BURST(1)
//)
//adapter
//(
//    .clk_i(clk_i),
//    .rst_i(arst_i),//

//    /*
//     * AXI slave interface
//     */
//    .s_axi_awid_i(s_axi_awid),
//    .s_axi_awaddr_i(s_axi_awaddr),
//    .s_axi_awlen_i(s_axi_awlen),
//    .s_axi_awsize_i(s_axi_awsize),
//    .s_axi_awburst_i(s_axi_awburst),
//    .s_axi_awlock_i(s_axi_awlock),
//    .s_axi_awcache_i(s_axi_awcache),
//    .s_axi_awprot_i(s_axi_awprot),
//    .s_axi_awqos_i(s_axi_awqos),
//    .s_axi_awregion_i(s_axi_awregion),
//    .s_axi_awuser_i(s_axi_awuser),
//    .s_axi_awvalid_i(s_axi_awvalid),
//    .s_axi_awready_o(s_axi_awready),
//    .s_axi_wdata_i(s_axi_wdata),
//    .s_axi_wstrb_i(s_axi_wstrb),
//    .s_axi_wlast_i(s_axi_wlast),
//    .s_axi_wuser_i(s_axi_wuser),
//    .s_axi_wvalid_i(s_axi_wvalid),
//    .s_axi_wready_o(s_axi_wready),
//    .s_axi_bid_o(s_axi_bid),
//    .s_axi_bresp_o(s_axi_bresp),
//    .s_axi_buser_o(s_axi_buser),
//    .s_axi_bvalid_o(s_axi_bvalid),
//    .s_axi_bready_i(s_axi_bready),
//    .s_axi_arid_i(s_axi_arid),
//    .s_axi_araddr_i(s_axi_araddr),
//    .s_axi_arlen_i(s_axi_arlen),
//    .s_axi_arsize_i(s_axi_arsize),
//    .s_axi_arburst_i(s_axi_arburst),
//    .s_axi_arlock_i(s_axi_arlock),
//    .s_axi_arcache_i(s_axi_arcache),
//    .s_axi_arprot_i(s_axi_arprot),
//    .s_axi_arqos_i(s_axi_arqos),
//    .s_axi_arregion_i(s_axi_arregion),
//    .s_axi_aruser_i(s_axi_aruser),
//    .s_axi_arvalid_i(s_axi_arvalid),
//    .s_axi_arready_o(s_axi_arready),
//    .s_axi_rid_o(s_axi_rid),
//    .s_axi_rdata_o(s_axi_rdata),
//    .s_axi_rresp_o(s_axi_rresp),
//    .s_axi_rlast_o(s_axi_rlast),
//    .s_axi_ruser_o(s_axi_ruser),
//    .s_axi_rvalid_o(s_axi_rvalid),
//    .s_axi_rready_i(s_axi_rready),//

//    /*
//     * AXI master interface
//     */
//    .m_axi_awid_o(m_axi_awid),
//    .m_axi_awaddr_o(m_axi_awaddr),
//    .m_axi_awlen_o(m_axi_awlen),
//    .m_axi_awsize_o(m_axi_awsize),
//    .m_axi_awburst_o(m_axi_awburst),
//    .m_axi_awlock_o(m_axi_awlock),
//    .m_axi_awcache_o(m_axi_awcache),
//    .m_axi_awprot_o(m_axi_awprot),
//    .m_axi_awqos_o(m_axi_awqos),
//    .m_axi_awregion_o(m_axi_awregion),
//    .m_axi_awuser_o(m_axi_awuser),
//    .m_axi_awvalid_o(m_axi_awvalid),
//    .m_axi_awready_i(m_axi_awready),
//    .m_axi_wdata_o(m_axi_wdata),
//    .m_axi_wstrb_o(m_axi_wstrb),
//    .m_axi_wlast_o(m_axi_wlast),
//    .m_axi_wuser_o(m_axi_wuser),
//    .m_axi_wvalid_o(m_axi_wvalid),
//    .m_axi_wready_i(m_axi_wready),
//    .m_axi_bid_i(m_axi_bid),
//    .m_axi_bresp_i(m_axi_bresp),
//    .m_axi_buser_i(m_axi_buser),
//    .m_axi_bvalid_i(m_axi_bvalid),
//    .m_axi_bready_o(m_axi_bready),
//    .m_axi_arid_o(m_axi_arid),
//    .m_axi_araddr_o(m_axi_araddr),
//    .m_axi_arlen_o(m_axi_arlen),
//    .m_axi_arsize_o(m_axi_arsize),
//    .m_axi_arburst_o(m_axi_arburst),
//    .m_axi_arlock_o(m_axi_arlock),
//    .m_axi_arcache_o(m_axi_arcache),
//    .m_axi_arprot_o(m_axi_arprot),
//    .m_axi_arqos_o(m_axi_arqos),
//    .m_axi_arregion_o(m_axi_arregion),
//    .m_axi_aruser_o(m_axi_aruser),
//    .m_axi_arvalid_o(m_axi_arvalid),
//    .m_axi_arready_i(m_axi_arready),
//    .m_axi_rid_i(m_axi_rid),
//    .m_axi_rdata_i(m_axi_rdata),
//    .m_axi_rresp_i(m_axi_rresp),
//    .m_axi_rlast_i(m_axi_rlast),
//    .m_axi_ruser_i(m_axi_ruser),
//    .m_axi_rvalid_i(m_axi_rvalid),
//    .m_axi_rready_o(m_axi_rready)
//);

endmodule
