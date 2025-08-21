#!/usr/bin/env python3

import argparse
from dataclasses import dataclass, field
from pathlib import Path
import subprocess


def parse_arguments():
    parser = argparse.ArgumentParser(
        prog="verilog_lint.py",
        description="""Verilog lint script.
        Run this tool to lint verilog over multiple modules.
        Currently supports verilator lint.""",
    )
    parser.add_argument(
        "-d",
        "--dir",
        action="append",
        default=None,
        help="Directory to search for Verilog files. Defaults to current directory.",
    )
    parser.add_argument(
        "-c",
        "--config",
        action="append",
        default=None,
        help="""Path to config files.
        Searches for verilator_config.vlt and [module]_waiver.vlt files.
        """,
    )
    parser.add_argument(
        "-o",
        "--output",
        default="verilog_lint.rpt",
        help="Output report file.",
    )

    args = parser.parse_args()

    # post process
    if not args.dir:
        args.dir = ["."]

    return parser.parse_args()


@dataclass
class VerilogModule:
    """Represent a Verilog module to lint"""
    name: str
    configs: list[str] = field(default_factory=list)


def get_verilog_modules(dirs: list[str]) -> list[VerilogModule]:
    """Get all verilog modules in directories.
    Args:
        dirs (list[str]): List of directories to search for Verilog files.
    Returns:
        list[VerilogModule]: List of Verilog module names.
    """
    vlog_modules: list = []
    for dir in dirs:
        # search for verilog modules in all *.v files
        matches = subprocess.run(
            f'grep "^module " {dir}/*.v',
            shell=True,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        for m in matches.splitlines():
            # grep output format:
            # ./path/to/file.v:module [module_name] #(
            name = m.split(":", 1)[1].split()[1]
            vlog_modules.append(
                VerilogModule(
                    name=name,
                )
            )
    return vlog_modules


def set_verilator_configs(vlog_modules: list[VerilogModule], cfg_dirs: list[str]) -> None:
    """Set verilator configs for each module.
    Update VerilogModules with respective configs.
    Args:
        vlog_modules (list[VerilogModule]): List of Verilog modules.
        cfg_dirs (list[str]): List of directories to search for config files.
    """
    for module in vlog_modules:
        for cfg_dir in cfg_dirs:
            # Search for verilator_config.vlt
            module.configs += [str(c) for c in list(Path(cfg_dir).glob("verilator_config.vlt"))]
            # Search for [module]_waiver.vlt configurations
            module.configs += [str(c) for c in list(Path(cfg_dir).glob(f"{module.name}_waiver.vlt"))]


def lint_modules(vlog_modules: list[VerilogModule], dirs: list[str]) -> None:
    """Lint each Verilog module using verilator.
    Update VerilogModule with lint results.
    Args:
        vlog_modules (list[VerilogModule]): List of Verilog modules to lint.
        dirs (list[str]): List of directories to search for Verilog files.
    """
    for module in vlog_modules:
        # run verilator lint command
        lint_cmd = "verilator --lint-only"
        lint_cmd += f" --top-module {module.name}"
        for cfg in module.configs:
            lint_cmd += f" {cfg}"
        for src_dir in dirs:
            lint_cmd += f" {src_dir}/*.v"
        print(f"Running lint command:\n\t{lint_cmd}\n\n")
        # TODO: actually run the lint command
        # Think about output
        # _ = subprocess.run(
        #     lint_cmd,
        #     shell=True,
        #     check=True,
        #     capture_output=True,
        #     text=True,
        # )
        # for m in matches.splitlines():
        #     # grep output format:
        #     # ./path/to/file.v:module [module_name] #(
        #     path, m_str = m.split(":", 1)
        #     name = m_str.split()[1]
        #     vlog_modules.append(
        #         VerilogModule(
        #             name=name,
        #             path=path,
        #         )
        #     )


if __name__ == "__main__":
    print("=====================")
    print("Verilog Linter Script")
    print("=====================\n")
    args = parse_arguments()

    # 1. Get all modules in dirs
    vlog_modules = get_verilog_modules(args.dir)
    # 2. Set verilator configs for each module
    set_verilator_configs(vlog_modules, args.config)
    # TODO:
    # 3. Run verilator lint on each module
    lint_modules(vlog_modules, args.dir)
    # 4. Process results into report file
    print(args)
