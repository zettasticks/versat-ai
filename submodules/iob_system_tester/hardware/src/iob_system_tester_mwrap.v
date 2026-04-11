// SPDX-FileCopyrightText: 2026 IObundle, Lda
//
// SPDX-License-Identifier: MIT
//
// Py2HWSW Version 0.81.0 has generated this code (https://github.com/IObundle/py2hwsw).

`timescale 1ns / 1ps
`include "iob_system_tester_mwrap_conf.vh"

module iob_system_tester_mwrap #(
    parameter AXI_ID_W = `IOB_SYSTEM_TESTER_MWRAP_AXI_ID_W,  // Don't change this parameter value!
    parameter AXI_ADDR_W = `IOB_SYSTEM_TESTER_MWRAP_AXI_ADDR_W,  // Don't change this parameter value!
    parameter AXI_DATA_W = `IOB_SYSTEM_TESTER_MWRAP_AXI_DATA_W,  // Don't change this parameter value!
    parameter AXI_LEN_W = `IOB_SYSTEM_TESTER_MWRAP_AXI_LEN_W,  // Don't change this parameter value!
    parameter SIMULATION = `IOB_SYSTEM_TESTER_MWRAP_SIMULATION,
    parameter ETH_PHY_RST_CNT = `IOB_SYSTEM_TESTER_MWRAP_ETH_PHY_RST_CNT,  // Don't change this parameter value!
    parameter BOOTROM_MEM_HEXFILE = `IOB_SYSTEM_TESTER_MWRAP_BOOTROM_MEM_HEXFILE,  // Don't change this parameter value!
    parameter INT_MEM_HEXFILE = `IOB_SYSTEM_TESTER_MWRAP_INT_MEM_HEXFILE,  // Don't change this parameter value!
    parameter MEM_NO_READ_ON_WRITE = `IOB_SYSTEM_TESTER_MWRAP_MEM_NO_READ_ON_WRITE
) (
    // clk_en_rst_s: Clock, clock enable and reset
    input clk_i,
    input cke_i,
    input arst_i,
    // axi_m: AXI manager interface for DDR memory
    output [AXI_ADDR_W-1:0] axi_araddr_o,
    output axi_arvalid_o,
    input axi_arready_i,
    input [AXI_DATA_W-1:0] axi_rdata_i,
    input [2-1:0] axi_rresp_i,
    input axi_rvalid_i,
    output axi_rready_o,
    output [AXI_ID_W-1:0] axi_arid_o,
    output [AXI_LEN_W-1:0] axi_arlen_o,
    output [3-1:0] axi_arsize_o,
    output [2-1:0] axi_arburst_o,
    output axi_arlock_o,
    output [4-1:0] axi_arcache_o,
    output [4-1:0] axi_arqos_o,
    input [AXI_ID_W-1:0] axi_rid_i,
    input axi_rlast_i,
    output [AXI_ADDR_W-1:0] axi_awaddr_o,
    output axi_awvalid_o,
    input axi_awready_i,
    output [AXI_DATA_W-1:0] axi_wdata_o,
    output [AXI_DATA_W/8-1:0] axi_wstrb_o,
    output axi_wvalid_o,
    input axi_wready_i,
    input [2-1:0] axi_bresp_i,
    input axi_bvalid_i,
    output axi_bready_o,
    output [AXI_ID_W-1:0] axi_awid_o,
    output [AXI_LEN_W-1:0] axi_awlen_o,
    output [3-1:0] axi_awsize_o,
    output [2-1:0] axi_awburst_o,
    output axi_awlock_o,
    output [4-1:0] axi_awcache_o,
    output [4-1:0] axi_awqos_o,
    output axi_wlast_o,
    input [AXI_ID_W-1:0] axi_bid_i,
    // rs232_m: iob-system uart interface
    input rs232_rxd_i,
    output rs232_txd_o,
    output rs232_rts_o,
    input rs232_cts_i,
    // phy_rstn_o: 
    output phy_rstn_o,
    // mii_io: Ethernet MII interface
    input mii_tx_clk_i,
    output [4-1:0] mii_txd_o,
    output mii_tx_en_o,
    output mii_tx_er_o,
    input mii_rx_clk_i,
    input [4-1:0] mii_rxd_i,
    input mii_rx_dv_i,
    input mii_rx_er_i,
    input mii_crs_i,
    input mii_col_i,
    inout mii_mdio_io,
    output mii_mdc_o
);

// Ports for connection with boot ROM memory
    wire bootrom_mem_clk;
    wire [10-1:0] bootrom_mem_addr;
    wire bootrom_mem_en;
    wire [32-1:0] bootrom_mem_r_data;
// Port for connection to 'iob_ram_t2p_be' memory
    wire int_mem_clk;
    wire int_mem_r_en;
    wire [28-1:0] int_mem_r_addr;
    wire [32-1:0] int_mem_r_data;
    wire [32/8-1:0] int_mem_w_strb;
    wire [28-1:0] int_mem_w_addr;
    wire [32-1:0] int_mem_w_data;

        // Wrapped module
        iob_system_tester #(
        .AXI_ID_W(AXI_ID_W),
        .AXI_ADDR_W(AXI_ADDR_W),
        .AXI_DATA_W(AXI_DATA_W),
        .AXI_LEN_W(AXI_LEN_W),
        .SIMULATION(SIMULATION),
        .ETH_PHY_RST_CNT(ETH_PHY_RST_CNT),
        .BOOTROM_MEM_HEXFILE(BOOTROM_MEM_HEXFILE),
        .INT_MEM_HEXFILE(INT_MEM_HEXFILE)
    ) iob_system_tester_inst (
            // rom_bus_m port: Ports for connection with boot ROM memory
        .bootrom_mem_clk_o(bootrom_mem_clk),
        .bootrom_mem_addr_o(bootrom_mem_addr),
        .bootrom_mem_en_o(bootrom_mem_en),
        .bootrom_mem_r_data_i(bootrom_mem_r_data),
        // int_mem_bus_m port: Port for connection to 'iob_ram_t2p_be' memory
        .int_mem_clk_o(int_mem_clk),
        .int_mem_r_en_o(int_mem_r_en),
        .int_mem_r_addr_o(int_mem_r_addr),
        .int_mem_r_data_i(int_mem_r_data),
        .int_mem_w_strb_o(int_mem_w_strb),
        .int_mem_w_addr_o(int_mem_w_addr),
        .int_mem_w_data_o(int_mem_w_data),
        // clk_en_rst_s port: Clock, clock enable and reset
        .clk_i(clk_i),
        .cke_i(cke_i),
        .arst_i(arst_i),
        // axi_m port: AXI manager interface for DDR memory
        .axi_araddr_o(axi_araddr_o),
        .axi_arvalid_o(axi_arvalid_o),
        .axi_arready_i(axi_arready_i),
        .axi_rdata_i(axi_rdata_i),
        .axi_rresp_i(axi_rresp_i),
        .axi_rvalid_i(axi_rvalid_i),
        .axi_rready_o(axi_rready_o),
        .axi_arid_o(axi_arid_o),
        .axi_arlen_o(axi_arlen_o),
        .axi_arsize_o(axi_arsize_o),
        .axi_arburst_o(axi_arburst_o),
        .axi_arlock_o(axi_arlock_o),
        .axi_arcache_o(axi_arcache_o),
        .axi_arqos_o(axi_arqos_o),
        .axi_rid_i(axi_rid_i),
        .axi_rlast_i(axi_rlast_i),
        .axi_awaddr_o(axi_awaddr_o),
        .axi_awvalid_o(axi_awvalid_o),
        .axi_awready_i(axi_awready_i),
        .axi_wdata_o(axi_wdata_o),
        .axi_wstrb_o(axi_wstrb_o),
        .axi_wvalid_o(axi_wvalid_o),
        .axi_wready_i(axi_wready_i),
        .axi_bresp_i(axi_bresp_i),
        .axi_bvalid_i(axi_bvalid_i),
        .axi_bready_o(axi_bready_o),
        .axi_awid_o(axi_awid_o),
        .axi_awlen_o(axi_awlen_o),
        .axi_awsize_o(axi_awsize_o),
        .axi_awburst_o(axi_awburst_o),
        .axi_awlock_o(axi_awlock_o),
        .axi_awcache_o(axi_awcache_o),
        .axi_awqos_o(axi_awqos_o),
        .axi_wlast_o(axi_wlast_o),
        .axi_bid_i(axi_bid_i),
        // rs232_m port: iob-system uart interface
        .rs232_rxd_i(rs232_rxd_i),
        .rs232_txd_o(rs232_txd_o),
        .rs232_rts_o(rs232_rts_o),
        .rs232_cts_i(rs232_cts_i),
        .phy_rstn_o(phy_rstn_o),
        // mii_io port: Ethernet MII interface
        .mii_tx_clk_i(mii_tx_clk_i),
        .mii_txd_o(mii_txd_o),
        .mii_tx_en_o(mii_tx_en_o),
        .mii_tx_er_o(mii_tx_er_o),
        .mii_rx_clk_i(mii_rx_clk_i),
        .mii_rxd_i(mii_rxd_i),
        .mii_rx_dv_i(mii_rx_dv_i),
        .mii_rx_er_i(mii_rx_er_i),
        .mii_crs_i(mii_crs_i),
        .mii_col_i(mii_col_i),
        .mii_mdio_io(mii_mdio_io),
        .mii_mdc_o(mii_mdc_o)
        );

            // Default description
        iob_rom_sp #(
        .DATA_W(32),
        .ADDR_W(10),
        .HEXFILE(BOOTROM_MEM_HEXFILE)
    ) bootrom_mem_mem (
            // rom_sp_s port: ROM interface
        .clk_i(bootrom_mem_clk),
        .addr_i(bootrom_mem_addr),
        .en_i(bootrom_mem_en),
        .r_data_o(bootrom_mem_r_data)
        );

            // Default description
        iob_ram_t2p_be #(
        .DATA_W(32),
        .ADDR_W(16),
        .HEXFILE(INT_MEM_HEXFILE)
    ) int_mem_mem (
            // ram_t2p_be_s port: RAM interface
        .clk_i(int_mem_clk),
        .r_en_i(int_mem_r_en),
        .r_addr_i(int_mem_r_addr[15:0]),
        .r_data_o(int_mem_r_data),
        .w_strb_i(int_mem_w_strb),
        .w_addr_i(int_mem_w_addr[15:0]),
        .w_data_i(int_mem_w_data)
        );

    
endmodule
