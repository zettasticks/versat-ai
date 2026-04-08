# SPDX-FileCopyrightText: 2025 IObundle
#
# SPDX-License-Identifier: MIT


def setup(py_params: dict):
    mem_addr_w = 25
    system_w = mem_addr_w
    name = "versat_ai"
    addr_w = 32
    data_w = 32

    # Set new default values for python parameters of iob_system (parent module)
    # List of iob_system python parameters available at: https://github.com/IObundle/py2hwsw/blob/main/py2hwsw/lib/iob_system/iob_system.py
    iob_system_default_overrides = {
        "init_mem": False,
        "use_intmem": False,
        "use_extmem": True,
        "use_ethernet": False,
        "mem_addr_w": mem_addr_w,
        "cpu": "iob_vexriscv",
        "fw_addr_w": mem_addr_w,
        # Tester configuration
        "include_tester": False,
        "tester_use_ethernet": True,
    }

    py_params = update_params(iob_system_default_overrides, py_params)

    # NOTE: Two configurations for this SUT core:
    # 1) SUT only (versat-ai without tester):
    #  - SUT uses ethernet
    #  - SUT does not have (external) memory initialized
    # 2) SUT+Tester (versat-ai and tester):
    #  - SUT does not use ethernet
    #  - SUT assumes external memory is initialized (tester handles initialization)

    if py_params["include_tester"]:
        py_params["use_ethernet"] = False  # No ethernet between SUT and Tester
        py_params["init_mem"] = True  # Tester initializes memory

    # setup custom xbar to include more subordinates (versat core)
    xbar_subblock = {
        "core_name": "iob_axi_full_xbar",
        "name": f"{name}_axi_full_xbar",
        "instance_name": "iob_axi_full_xbar",
        "instance_description": "AXI full xbar instance",
        "parameters": {
            "ID_W": "AXI_ID_W",
            "LEN_W": "AXI_LEN_W",
        },
        "connect": {
            "clk_en_rst_s": "clk_en_rst_s",
            "rst_i": "rst",
            "s0_axi_s": "cpu_ibus",
            "s1_axi_s": "cpu_dbus",
            # "s2_axi_s": "versat_axi",
            # Manager interfaces connected below
        },
        "addr_w": addr_w,
        "data_w": data_w,
        "lock_w": 1,
        "num_subordinates": 3,
    }
    # Add ethernet connections in xbar subblock if needed
    if py_params["use_ethernet"]:
        subordinate_if_number = xbar_subblock["num_subordinates"]
        xbar_subblock["num_subordinates"] += 1
        xbar_subblock["connect"] |= {
            f"s{subordinate_if_number}_axi_s": (
                "eth_axi",
                ["eth_axi_awlock[0]", "eth_axi_arlock[0]"],
            )
        }
    xbar_manager_interfaces = {
        "use_extmem": (
            "axi_m",
            [
                "{unused_m1_araddr_bits, axi_araddr_o}",
                "{unused_m1_awaddr_bits, axi_awaddr_o}",
            ],
        ),
        "use_bootrom": (
            "bootrom_cbus",
            [
                "{unused_m2_araddr_bits, bootrom_axi_araddr}",
                "{unused_m2_awaddr_bits, bootrom_axi_awaddr}",
            ],
        ),
        "use_peripherals": (
            "axi_periphs_cbus",
            [
                "{unused_m3_araddr_bits, periphs_axi_araddr}",
                "{unused_m3_awaddr_bits, periphs_axi_awaddr}",
                "periphs_axi_awlock[0]",
                "periphs_axi_arlock[0]",
            ],
        ),
    }
    # Connect xbar manager interfaces
    num_managers = 0
    for interface_connection in xbar_manager_interfaces.values():
        xbar_subblock["connect"] |= {f"m{num_managers}_axi_m": interface_connection}
        num_managers += 1
    xbar_subblock["num_managers"] = num_managers

    attributes_dict = {
        "title": "Versat-AI System",
        "description": "Accelerate AI Applications with Versat-AI.",
        "board_list": [
            "iob_aes_ku040_db_g"
            # "iob_cyclonev_gt_dk",
            # "iob_zybo_z7",
        ],
        "confs": [
            # {   TODO: is this needed?
            #     "name": "INT_MEM_HEXFILE",
            #     "descr": "Firmware file name",
            #     "type": "D",
            #     "val": f'"{name}_firmware"',  # NOTE: The '"' inside are on purpose
            #     "min": "NA",
            #     "max": "NA",
            # },
            {
                "name": "EXT_MEM_HEXFILE",
                "descr": "Firmware file name",
                "type": "D",
                "val": f'"{name}_firmware"',  # NOTE: The '"' inside are on purpose
                "min": "NA",
                "max": "NA",
            },
        ],
        "ports": [
            {
                # Add new rs232 port for uart
                "name": "rs232_m",
                "descr": "iob-system uart interface",
                "signals": {
                    "type": "rs232",
                },
            },
            {
                "name": "csrs_cbus_s",
                "descr": "Control/Status Registers of versat-ai system (using regfileif).",
                "signals": {
                    "type": "iob",
                    "ADDR_W": 3,
                    "DATA_W": data_w,
                },
            },
            # NOTE: Add other ports here.
        ],
        "wires": [
            # {
            #    "name": "versat_axi",
            #    "descr": "Versat axi wires",
            #    "signals": {
            #        "type": "axi",
            #        "prefix": "versat_",
            #        "ID_W": "AXI_ID_W",
            #        "ADDR_W": addr_w,
            #        "DATA_W": data_w,
            #        "LEN_W": "AXI_LEN_W",
            #        "LOCK_W": "1",
            #    },
            # },
            # CPU control wires
            {"name": "rst", "signals": [{"name": "sw_reset", "width": 1}]},
            # {  # FIXME: Connect this to CPU reset addr (or use preboot to jump to correct one) (or use address translator in tester).
            #     "name": "fw_base_addr",
            #     "signals": [{"name": "fw_base_addr", "width": addr_w}],
            # },
        ],
        "subblocks": [
            xbar_subblock,
            {
                # Instantiate a UART core from: https://github.com/IObundle/py2hwsw/tree/main/py2hwsw/lib/hardware/iob_uart
                "core_name": "iob_uart",
                "instance_name": "UART0",
                "instance_description": "UART peripheral",
                "is_peripheral": True,
                "parameters": {},
                "connect": {
                    "clk_en_rst_s": "clk_en_rst_s",
                    # Cbus connected automatically
                    "rs232_m": "rs232_m",
                },
            },
            {
                # Instantiate a TIMER core from: https://github.com/IObundle/py2hwsw/tree/main/py2hwsw/lib/hardware/iob_timer
                "core_name": "iob_timer",
                "instance_name": "TIMER0",
                "instance_description": "Timer peripheral",
                "is_peripheral": True,
                "parameters": {},
                "connect": {
                    "clk_en_rst_s": "clk_en_rst_s",
                    # Cbus connected automatically
                },
            },
            # {
            #    "core_name": "iob_versat",
            #    "instance_name": "VERSAT0",
            #    "instance_description": "Versat accelerator",
            #    "is_peripheral": True,
            #    "parameters": {},
            #    "connect": {
            #        "clk_en_rst_s": "clk_en_rst_s",
            #        "axi_out_m": "versat_axi",
            #        # Cbus connected automatically
            #    },
            # },
            {
                "core_name": "iob_regfileif",
                "instance_name": "REGFILEIF0",
                "instance_description": "Provides Register file interface with registers used to configure, control and monitor the Versat system.",
                "is_peripheral": True,
                "internal_csr_if_widths": {
                    "ADDR_W": 3,
                    "DATA_W": 32,
                },
                "external_csr_if_widths": {
                    "ADDR_W": 3,
                    "DATA_W": 32,
                },
                "csrs": [
                    {
                        "name": "regfileif",
                        "descr": "REGFILEIF software accessible registers.",
                        "regs": [
                            {
                                "name": "start",
                                "descr": "Set 1 to start. versat_ai sets this to 0 to acknowledge start received",
                                "mode": "RW",
                                "n_bits": 1,
                                "rst_val": 0,
                                "log2n_items": 0,
                            },
                            {
                                "name": "done",
                                "descr": "Set to 1 by versat_ai when finished. 0 while running",
                                "mode": "R",
                                "n_bits": 1,
                                "rst_val": 0,
                                "log2n_items": 0,
                            },
                            {
                                "name": "metamodel_addr",
                                "descr": "Metamodel address",
                                "mode": "W",
                                "n_bits": 32,
                                "rst_val": 0,
                                "log2n_items": 0,
                            },
                            {
                                "name": "output_addr",
                                "descr": "Model address",
                                "mode": "W",
                                "n_bits": 32,
                                "rst_val": 0,
                                "log2n_items": 0,
                            },
                            {
                                "name": "temp_addr",
                                "descr": "Model address",
                                "mode": "W",
                                "n_bits": 32,
                                "rst_val": 0,
                                "log2n_items": 0,
                            },
                            {
                                "name": "model_addr",
                                "descr": "Model address",
                                "mode": "W",
                                "n_bits": 32,
                                "rst_val": 0,
                                "log2n_items": 0,
                            },
                            {
                                "name": "correctOutputs_addr",
                                "descr": "Correct output address",
                                "mode": "W",
                                "n_bits": 32,
                                "rst_val": 0,
                                "log2n_items": 0,
                            },
                            {
                                "name": "inputsVector_addr",
                                "descr": "Inputs address",
                                "mode": "W",
                                "n_bits": 32,
                                "rst_val": 0,
                                "log2n_items": 0,
                            },
                            # Versat-ai CPU control registers
                            {
                                "name": "rst",
                                "descr": "Resets CPU (1) or not (0).",
                                "mode": "W",
                                "n_bits": 1,
                                "rst_val": 0,  # Set to 1 to prevent versat from booting at startup
                                "log2n_items": 0,
                                "output": True,  # Generate dedicated output port with value of this CSR
                            },
                            # {
                            #     "name": "firm_addr",
                            #     "descr": "Memory address of Versat firmware were CPU boots from.",
                            #     "mode": "W",
                            #     "n_bits": 32,
                            #     "rst_val": 0,
                            #     "log2n_items": 0,
                            #     "output": True,  # Generate dedicated output port with value of this CSR
                            # },
                        ],
                    },
                ],
                "connect": {
                    "clk_en_rst_s": "clk_en_rst_s",
                    # Cbus connected automatically
                    "csrs_external_cbus_s": "csrs_cbus_s",
                    "rst_o": "rst",
                    # "firm_addr_o": "fw_base_addr",
                },
            },
            # NOTE: Add other components/peripherals here.
        ],
        "sw_modules": [
            {
                "core_name": "iob_coverage_analyze",
                "instance_name": "iob_coverage_analyze_inst",
            },
        ],
    }
    if py_params["include_tester"]:
        attributes_dict["superblocks"] = [
            {  # Override tester with this one to pass custom sut py parameters
                "core_name": "iob_system_tester",
                "instance_name": "iob_system_tester",
                "sut_py_params": py_params,
                "dest_dir": "tester",
            },
        ]

    core_dict = {
        "version": "0.8.0",
        "parent": {
            "core_name": "iob_system",
            "system_attributes": attributes_dict,
            **py_params,
        },
    }

    return core_dict


#
# Utility functions
#


def update_params(params, py_params):
    # Default parameters
    new_params = params.copy()
    # Process python parameters
    for name, override_val in py_params.items():
        # If py param is a new parameter, append it to dict
        if name not in params:
            new_params[name] = override_val
            continue

        # Otherwise, py param will override default one
        default_val = params[name]
        # If py_param corresponds to a bool, and matches string "0", convert it to False
        # This is needed because running in python: bool("0") == True
        if type(default_val) is bool and override_val == "0":
            new_params[name] = False
        # Otherwise, convert py_param to correct type (for example: bool(override_val), int(override_val), etc)
        else:
            new_params[name] = type(default_val)(override_val)

    return new_params
