# SPDX-FileCopyrightText: 2025 IObundle
#
# SPDX-License-Identifier: MIT


def setup(py_params_dict):
    params = py_params_dict["iob_system_params"]

    addr_w = params["mem_addr_w"]
    data_w = 32
    offset = 2

    # data_w = 64
    # offset = 3

    # Size of RAM for ethernet's dma
    ETH_RAM_ADDR_W = 14

    tb_peripherals = ["iob_uart"]
    if params["use_ethernet"]:
        tb_peripherals += ["iob_eth"]

    periph_sel_bits = (len(tb_peripherals) - 1).bit_length()
    periph_addr_w = 32 - periph_sel_bits

    attributes_dict = {
        "name": "iob_uut",
        "generate_hw": True,
        "confs": [
            {
                "name": "AXI_ID_W",
                "descr": "AXI ID bus width",
                "type": "D",
                "val": "4",
            },
            {
                "name": "AXI_LEN_W",
                "descr": "AXI burst length width",
                "type": "D",
                "val": "8",
            },
            {
                "name": "AXI_ADDR_W",
                "descr": "AXI address bus width",
                "type": "D",
                "val": params["mem_addr_w"],
            },
            {
                "name": "AXI_DATA_W",
                "descr": "AXI data bus width",
                "type": "D",
                "val": data_w,
            },
            {
                "name": "BAUD",
                "descr": "UART baud rate",
                "type": "D",
                "val": "3000000",
            },
            {
                "name": "FREQ",
                "descr": "Clock frequency",
                "type": "D",
                "val": "100000000",
            },
            {
                "name": "SIMULATION",
                "descr": "Simulation flag",
                "type": "D",
                "val": "1",
            },
        ],
    }

    #
    # Ports
    #
    attributes_dict["ports"] = [
        {
            "name": "clk_en_rst_s",
            "descr": "Clock, clock enable and reset",
            "signals": {
                "type": "iob_clk",
            },
        },
        {
            "name": "axi_m",
            "descr": "AXI manager interface for DDR memory",
            "signals": {
                "type": "axi",
                "ID_W": "AXI_ID_W",
                "ADDR_W": "AXI_ADDR_W",
                "DATA_W": data_w,
                "LEN_W": "AXI_LEN_W",
                "LOCK_W": 1,
            },
        },
        {
            "name": "tb_s",
            "descr": "Testbench iob interface",
            "signals": {
                "type": "iob",
                "ADDR_W": 32,
            },
        },
    ]

    #
    # Wires
    #
    attributes_dict["wires"] = [
        {
            "name": "clk",
            "descr": "Clock signal",
            "signals": [
                {"name": "clk_i"},
            ],
        },
        {
            "name": "rst",
            "descr": "Reset signal",
            "signals": [
                {"name": "arst_i"},
            ],
        },
        {
            "name": "rs232",
            "descr": "rs232 bus",
            "signals": {
                "type": "rs232",
            },
        },
        {
            "name": "rs232_invert",
            "descr": "Invert order of rs232 signals",
            "signals": [
                {"name": "rs232_txd"},
                {"name": "rs232_rxd"},
                {"name": "rs232_cts"},
                {"name": "rs232_rts"},
            ],
        },
    ]
    if len(tb_peripherals) > 1:
        attributes_dict["wires"] += [
            {
                "name": "uart_cbus",
                "descr": "UART CSR bus",
                "signals": {
                    "type": "iob",
                    "prefix": "uart_",
                    "ADDR_W": periph_addr_w,
                },
            },
        ]
    if params["use_extmem"]:
        attributes_dict["wires"] += [
            {
                "name": "uut_axi",
                "descr": "AXI bus to connect SoC to interconnect",
                "signals": {
                    "type": "axi",
                    "prefix": "uut_",
                    "ID_W": "AXI_ID_W",
                    "ADDR_W": "AXI_ADDR_W",
                    "DATA_W": data_w,
                    "LEN_W": "AXI_LEN_W",
                    "LOCK_W": 1,
                },
            },
            {
                "name": "axi_ram_mem",
                "descr": "Connect axi_ram to 'iob_ram_t2p_be' memory",
                "signals": {
                    "type": "ram_t2p_be",
                    "prefix": "ext_mem_",
                    "ADDR_W": "AXI_ADDR_W - 2",
                    "DATA_W": data_w,
                },
            },
        ]

    #
    # Blocks
    #
    attributes_dict["subblocks"] = [
        {
            "core_name": py_params_dict["issuer"]["original_name"],
            "instance_name": py_params_dict["issuer"]["original_name"],
            "instance_description": "IOb-SoC memory wrapper",
            "parameters": {
                "AXI_ID_W": "AXI_ID_W",
                "AXI_LEN_W": "AXI_LEN_W",
                "AXI_ADDR_W": "AXI_ADDR_W",
                "AXI_DATA_W": 32,
                "SIMULATION": "SIMULATION",
            },
            "connect": {
                "clk_en_rst_s": "clk_en_rst_s",
                "rs232_m": "rs232",
                "axi_m": "uut_axi",
            },
            "dest_dir": "hardware/common_src",
        },
    ]

    # Connect ethernet and its RAM to pbus
    attributes_dict["subblocks"] += [
        {
            "core_name": "iob_uart",
            "instance_name": "uart_tb",
            "instance_description": "Testbench uart core",
            "csr_if": "iob",
            "connect": {
                "clk_en_rst_s": "clk_en_rst_s",
                "csrs_cbus_s": ("uart_cbus", ["uart_iob_addr[3:0]"]),
                "rs232_m": "rs232_invert",
            },
        },
    ]

    if len(tb_peripherals) == 1:
        # Connect uart directly to tb_s port if there is no tb_pbus_split
        attributes_dict["subblocks"][-1]["connect"].update(
            {"csrs_cbus_s": ("tb_s", ["iob_addr_i[3:0]"])}
        )

    if params["use_extmem"]:
        attributes_dict["subblocks"] += [
            {
                "core_name": "iob_axi_ram",
                "instance_name": "ddr_model_mem",
                "instance_description": "External memory",
                "parameters": {
                    "ID_WIDTH": "AXI_ID_W",
                    "ADDR_WIDTH": "AXI_ADDR_W",
                    "DATA_WIDTH": data_w,
                },
                "connect": {
                    "clk_i": "clk",
                    "rst_i": "rst",
                    "axi_s": (
                        "uut_axi",
                        [
                            "{1'b0, uut_axi_arlock}",
                            "{1'b0, uut_axi_awlock}",
                        ],
                    ),
                    "external_mem_bus_m": "axi_ram_mem",
                },
            },
            {
                "core_name": "iob_ram_t2p_be",
                "instance_name": "iob_ram_t2p_be_inst",
                "parameters": {
                    "ADDR_W": f"AXI_ADDR_W - {offset}",
                    "DATA_W": data_w,
                },
                "connect": {
                    "ram_t2p_be_s": "axi_ram_mem",
                },
            },
        ]
        if params["init_mem"] and not params["use_intmem"]:
            attributes_dict["subblocks"][-1]["parameters"].update(
                {
                    "HEXFILE": f'"{params["name"]}_firmware"',
                }
            )
    #
    # Snippets
    #
    attributes_dict["snippets"] = []

    # Calculate and print testbench peripheral memory map
    print("------------------------------------------------------")
    print("Testbench memory map:")
    current_addr = 0
    for peripheral in tb_peripherals:
        print(
            f"[0x{current_addr:08x}-0x{(current_addr+(1<<periph_addr_w)-1):08x}]: {peripheral} ({periph_addr_w} bits)"
        )
        current_addr += 1 << periph_addr_w
    print("------------------------------------------------------")

    return attributes_dict
