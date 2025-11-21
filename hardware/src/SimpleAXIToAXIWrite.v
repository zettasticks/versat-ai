`timescale 1ns / 1ps

// Since Simple interface expects to write N transfers regardless of anything else (does not care about alignment or axi boundary)
// it becomes simpler to just have a module that performs N transfers from 
module TransferNFromSimpleM #(
   parameter AXI_DATA_W = 32,
   parameter MAX_TRANSF_W = 32
   ) (
      input [MAX_TRANSF_W-1:0] transferCount_i,
      input                    initiateTransfer_i,

      // NOTE: Because axi is divided into bursts (meaning that the transfer is paused temporarely)
      //       while this unit is always working after starting, that means that we need to
      //       make sure that the signals that are connected to axi are not "enabled" 
      //       unless we are in a state where the transfer is ocurring. Basically, while axi is not
      //       using the W channel, the inputs of this unit and the outputs basically need to be set
      //       to zero.

      // Connect directly to simple axi 
      input                        m_wvalid_i,
      output                       m_wready_o,
      input [      AXI_DATA_W-1:0] m_wdata_i,
      output                       m_wlast_o,

      // Data output
      output                  data_valid_o,
      output [AXI_DATA_W-1:0] data_o,
      input                   data_ready_i,

      input rst_i,
      input clk_i
   );

reg [MAX_TRANSF_W-1:0] count;
reg working;

assign m_wready_o = working && data_ready_i;
assign data_valid_o = working && m_wvalid_i;
assign data_o = m_wdata_i;

assign m_wlast_o = working && (count <= 1);

always @(posedge clk_i,posedge rst_i) begin
   if(rst_i) begin
      count <= 0;
      working <= 1'b0;
   end else if(working) begin
      if(m_wvalid_i && m_wready_o) begin
         count <= count - 1;

         if(m_wlast_o) begin
            working <= 1'b0;
         end
      end
   end else if(initiateTransfer_i) begin
      count <= transferCount_i;
      working <= 1'b1;
   end
end

endmodule

module SimpleAXItoAXIWrite #(
   parameter AXI_ADDR_W = 32,
   parameter AXI_DATA_W = 32,
   parameter AXI_LEN_W  = 8,
   parameter AXI_ID_W   = 1,
   parameter LEN_W      = 8
) (
   input                             m_wvalid_i,
   output                            m_wready_o,
   input      [      AXI_ADDR_W-1:0] m_waddr_i,
   input      [      AXI_DATA_W-1:0] m_wdata_i,
   input      [(AXI_DATA_W / 8)-1:0] m_wstrb_i,
   input      [           LEN_W-1:0] m_wlen_i,
   output                            m_wlast_o,

   output [AXI_ID_W-1:0] axi_awid_o,
   output [AXI_ADDR_W-1:0] axi_awaddr_o,
   output [AXI_LEN_W-1:0] axi_awlen_o,
   output [3-1:0] axi_awsize_o,
   output [2-1:0] axi_awburst_o,
   output [2-1:0] axi_awlock_o,
   output [4-1:0] axi_awcache_o,
   output [3-1:0] axi_awprot_o,
   output [4-1:0] axi_awqos_o,
   output [1-1:0] axi_awvalid_o,
   input [1-1:0] axi_awready_i,
   output [AXI_DATA_W-1:0] axi_wdata_o,
   output [(AXI_DATA_W/8)-1:0] axi_wstrb_o,
   output [1-1:0] axi_wlast_o,
   output [1-1:0] axi_wvalid_o,
   input [1-1:0] axi_wready_i,
   input [AXI_ID_W-1:0] axi_bid_i,
   input [2-1:0] axi_bresp_i,
   input [1-1:0] axi_bvalid_i,
   output [1-1:0] axi_bready_o,

   input clk_i,
   input rst_i
);

   reg [2:0] state;

   // This contains both the logic for the AXI transfer and the databus transfer.
   // We first need to decouple this before progressing.
    // Time == 0
   reg [31:0] totalTransferLength;
   reg [AXI_ADDR_W-1:0] address;

   // Time == 1
   reg [31:0] total_symbols_to_transfer;
   reg [8:0]  true_symbols_to_transfer;
   reg [7:0]  true_axi_awlen_temp;
   reg [7:0]  true_axi_awlen;

   // Depends on totalTransferLength
   always @* begin
      total_symbols_to_transfer = ((totalTransferLength - 1) >> 2) + 1;
      
      if(total_symbols_to_transfer[31:8] == 0) begin
         true_symbols_to_transfer = {1'b0,total_symbols_to_transfer[7:0]};
      end else begin
         true_symbols_to_transfer = 9'h100;
      end
      true_axi_awlen_temp = true_symbols_to_transfer - 1;
      true_axi_awlen = true_axi_awlen_temp;
   end

   wire [31:0] length_to_transfer = {21'b0,true_symbols_to_transfer,2'b00};
   reg [31:0] transferLengthChange;
   always @* begin
      transferLengthChange = length_to_transfer;
      if(length_to_transfer > totalTransferLength) begin
         transferLengthChange = totalTransferLength;
      end 
   end

   wire w_transfer = (state == 3);

   reg  first_transfer;
   wire transfer_valid_o;
   wire transfer_ready;

   assign transfer_ready = (axi_wready_i && w_transfer);
   assign axi_wvalid_o = (transfer_valid_o && w_transfer);

   //assign axi_wdata_o = totalTransferLength;

   TransferNFromSimpleM #(
      .AXI_DATA_W(AXI_DATA_W)
   ) transferN (
      .transferCount_i(total_symbols_to_transfer),
      .initiateTransfer_i(state == 2 && axi_awready_i && first_transfer),

      // Connect directly to simple axi 
      .m_wvalid_i(m_wvalid_i),
      .m_wready_o(m_wready_o),
      .m_wdata_i(m_wdata_i),
      .m_wlast_o(m_wlast_o),

      // Data output
      .data_valid_o(transfer_valid_o),
      .data_o(axi_wdata_o),
      .data_ready_i(transfer_ready),

      .rst_i(rst_i),
      .clk_i(clk_i)
   );

   // Address write constants
   assign axi_awid_o    = 0;
   assign axi_awsize_o  = 3'b010;
   assign axi_awburst_o = 2'b01;  // INCR
   assign axi_awlock_o  = 0;
   assign axi_awcache_o = 0;
   assign axi_awprot_o  = 0;
   assign axi_awqos_o   = 0;

   reg [3:0] axi_wstrb;
   assign axi_wstrb_o  = axi_wstrb;
   
   reg axi_bready;
   assign axi_bready_o = axi_bready;

   assign axi_awaddr_o = address;

   reg [AXI_LEN_W-1:0] awlen_reg;
   assign axi_awlen_o = awlen_reg;

   reg [AXI_LEN_W-1:0] counter;
   wire axi_last = (state == 3) && (counter >= awlen_reg);
   assign axi_wlast_o = axi_last;

   reg awvalid_reg;
   assign axi_awvalid_o = awvalid_reg;

   always @(posedge clk_i, posedge rst_i) begin
      if (rst_i) begin
         state               <= 0;
         awvalid_reg         <= 0;
         counter             <= 0;
         totalTransferLength <= 0;
         awlen_reg           <= 0;
         first_transfer      <= 0;
         axi_bready          <= 0;
         axi_wstrb           <= 0;
         address             <= 0;

      end else begin
         case (state)
            3'h0: begin  // Wait one cycle for transfer controller to calculate things.
               if (m_wvalid_i) begin
                  state               <= 3'h1;
                  totalTransferLength <= {24'b0,m_wlen_i};
                  address             <= m_waddr_i;
                  first_transfer      <= 1'b1;
               end
            end
            3'h1: begin  // Save values that change 
               awlen_reg   <= true_axi_awlen; // true_axi_awlen;
               awvalid_reg <= 1'b1;
               state       <= 3'h2;
            end
            3'h2: begin  // Write address set
               // awvalid is 1 at this point. Waiting for axi_awready_i
               if (axi_awready_i) begin
                  awvalid_reg   <= 1'b0;
                  state         <= 3'h3;
                  counter       <= 0;
                  axi_wstrb     <= 4'hf;

                  // Update next burst content in here
                  totalTransferLength <= totalTransferLength - transferLengthChange;
                  address <= address + transferLengthChange;
               end
            end
            3'h3: begin
               // Transfer, axi_wvalid set and waiting for axi_wready_i
               if (axi_wvalid_o && axi_wready_i) begin
                  counter <= counter + 1;

                  if (axi_last) begin
                     axi_wstrb <= 0;
                     axi_bready <= 1'b1;
                     state <= 3'h4;
                  end
               end
            end
            3'h4: begin
               if(axi_bvalid_i) begin
                  axi_bready <= 1'b0;

                  // Ony cycle delay between transfers because we could be hitting a bug where the interconnect takes an extra cycle switching masters. 
                  state <= 3'h5;
               end
            end
            3'h5: begin
               if (totalTransferLength == 0) begin
                  state <= 3'h0;
               end else begin
                  state <= 3'h1;
                  first_transfer <= 1'b0;
               end
            end
            default: begin
               state <= 3'h0;
            end
         endcase
      end
   end

endmodule  // SimpleAXItoAXIWrite
