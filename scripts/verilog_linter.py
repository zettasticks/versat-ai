#!/usr/bin/env python3

import argparse
from collections import deque
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
    parser.add_argument(
        "--gen-waiver",
        action="store_true",
        help="Generate [module]_waiver.vlt waiver file based on linter warnings.",
    )

    args = parser.parse_args()

    # post process
    if not args.dir:
        args.dir = ["."]

    return parser.parse_args()


@dataclass
class VerilogModule:
    """Represent a Verilog module to lint"""

    name: str = ""
    vfile: str = ""
    module_tree: list[str] = field(default_factory=list)
    configs: list[str] = field(default_factory=list)
    result: subprocess.CompletedProcess | None = None


def build_module_trees(vlog_modules: list[VerilogModule]) -> None:
    """Build module trees for each Verilog module.
    Read the module file and list the instantiated modules.
    Args:
        vlog_modules (list[VerilogModule]): List of Verilog modules.
    """
    for module in vlog_modules:
        with open(module.vfile, "r") as f:
            lines = f.readlines()
            for line in lines:
                if "#(" in line:
                    if line.startswith("module "):
                        # skip
                        continue
                    # line format:
                    # module_name #( ....
                    module.module_tree.append(line.split("#(")[0].strip())
        # remove duplicates
        module.module_tree = list(set(module.module_tree))


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
            vfile, m_str = m.split(":", 1)
            name = m_str.split()[1]
            vlog_modules.append(
                VerilogModule(
                    name=name,
                    vfile=vfile,
                )
            )

    build_module_trees(vlog_modules)
    for module in vlog_modules:
        print(f"Found module: {module.name} in {module.vfile}")
        print(module.module_tree)
        print()
    return vlog_modules


def dict_from_list(modules: list[VerilogModule]) -> dict[str, VerilogModule]:
    """Convert a list of VerilogModule to a dictionary.
    key = module name, value = VerilogModule object.
    Args:
        modules (list[VerilogModule]): List of Verilog modules.
    Returns:
    dict[str, VerilogModule]: Dictionary of Verilog modules.
    """
    d: dict = {}
    for m in modules:
        d[m.name] = m
    return d


def files_from_tree(top_module: VerilogModule, modules: dict[str, VerilogModule]) -> list[str]:
    """Get all files from module tree.
    Follows module tree from top_module and gets all files for submodules recursively.
    Args:
        top_module (VerilogModule): The Verilog module to get files from.
        modules (dict[str, VerilogModule]): List of all Verilog modules.
    Returns:
        list[str]: List of Verilog files in the module tree.
    """
    files: list[str] = []
    traverse = deque()
    traverse.append(top_module.name)
    while traverse:
        # get next module name
        m_name = traverse.popleft()
        # get module from name
        try:
            module = modules[m_name]
        except KeyError:
            # if module not found, skip
            continue
        # add vfile to list:
        files.append(module.vfile)
        # add submodules to traverse queue
        for submodule in module.module_tree:
            traverse.append(submodule)
    # remove duplicates
    return list(set(files))


def set_verilator_configs(
    vlog_modules: list[VerilogModule], cfg_dirs: list[str]
) -> None:
    """Set verilator configs for each module.
    Update VerilogModules with respective configs.
    Args:
        vlog_modules (list[VerilogModule]): List of Verilog modules.
        cfg_dirs (list[str]): List of directories to search for config files.
    """
    for module in vlog_modules:
        for cfg_dir in cfg_dirs:
            # Search for verilator_config.vlt
            module.configs += [
                str(c) for c in list(Path(cfg_dir).glob("verilator_config.vlt"))
            ]
            # Search for [module]_waiver.vlt configurations
            module.configs += [
                str(c) for c in list(Path(cfg_dir).glob(f"{module.name}_waiver.vlt"))
            ]


def lint_modules(
    vlog_modules: list[VerilogModule], dirs: list[str], gen_waiver: bool
) -> None:
    """Lint each Verilog module using verilator.
    Update VerilogModule with lint results.
    Args:
        vlog_modules (list[VerilogModule]): List of Verilog modules to lint.
        dirs (list[str]): List of directories to search for Verilog files.
    """
    mod_dict: dict[str, VerilogModule] = dict_from_list(vlog_modules)
    for module in vlog_modules:
        # run verilator lint command
        lint_cmd = "verilator --lint-only"
        if gen_waiver:
            waiver_name = f"{module.name}_waiver.vlt"
            lint_cmd += f" --waiver-output {waiver_name}"
        lint_cmd += f" --top-module {module.name}"
        for cfg in module.configs:
            lint_cmd += f" {cfg}"
        # add verilog source files
        vfiles = files_from_tree(module, mod_dict)
        for v in vfiles:
            lint_cmd += f" {v}"
        print(f"Running lint command:\n\t{lint_cmd}\n\n")
        module.result = subprocess.run(
            lint_cmd,
            shell=True,
            capture_output=True,
            text=True,
        )


def process_results(vlog_modules: list[VerilogModule], output: str) -> None:
    """Process lint results and write to output file.
    Args:
        vlog_modules (list[VerilogModule]): List of Verilog modules with lint results.
        output (str): Output report file path.
    """
    with open(output, "w") as f:
        passed = 0
        failed = 0
        total = 0
        for module in vlog_modules:
            if module.result.returncode == 0:
                passed += 1
            else:
                failed += 1
            total += 1
        f.write("// Report Generated by verilog_linter.py\n\n")
        f.write("====================\n")
        f.write("Lint Results Summary\n")
        f.write("====================\n")
        f.write(f"Modules lint passed: {passed}\n")
        f.write(f"Modules lint failed: {failed}\n")
        f.write(f"Total Modules linted: {total}\n\n")

        f.write("===================\n")
        f.write("Linted Modules List\n")
        f.write("===================\n")
        for module in vlog_modules:
            if module.result.returncode == 0:
                f.write(f"{module.name} - PASSED\n")
            else:
                f.write(f"{module.name} - FAILED\n")

        f.write("\n======================\n")
        f.write("Detailed Lint Warnings\n")
        f.write("======================\n")
        # Detailed Lint Warnings
        for module in vlog_modules:
            if module.result.returncode != 0:
                f.write(f"Module: {module.name}\n")
                f.write(f"Lint command: {module.result.args}\n")
                f.write(f"Warnings:\n{module.result.stderr}\n")
                f.write("\n=================================\n\n")


if __name__ == "__main__":
    print("=====================")
    print("Verilog Linter Script")
    print("=====================\n")
    args = parse_arguments()
    print(args)

    # 1. Get all modules in dirs
    vlog_modules = get_verilog_modules(args.dir)
    # 2. Set verilator configs for each module
    set_verilator_configs(vlog_modules, args.config)
    # 3. Run verilator lint on each module
    lint_modules(vlog_modules, args.dir, args.gen_waiver)
    # 4. Process results into report file
    process_results(vlog_modules, args.output)
