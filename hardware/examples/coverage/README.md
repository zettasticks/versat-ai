# Coverage Example
Standalone example for FU Coverage

## Contents:
hardware/examples/coverage/
├── Makefile
├── README.md
├── scripts
│   └── coverage_analyze.py
└── src
    ├── relu_tb.v
    └── Relu.v

### Run example:
```bash
# coverage with verilog only testbench
make cov-verilog-only
# optional lcov report
make gen-lcov-report
```

### Check results:
- `coverage` target output:
```bash
$ verilator_coverage --annotate cov_annotated --annotate-min 2 --annotate-all merged.dat
Total coverage (12/12) 100.00%
```
- manual check results in `cov_annotated/` directory
- more info about annotated results
  [here](https://verilator.org/guide/latest/exe_verilator_coverage.html)

## Optional Verilog Annotation Script
`coverage_analyze.py`: custom script that analyzes verilator coverage annotations.
- check results in `coverage.rpt`

## Optional graphical report:
- `cov-lcov` makefile target
    - requires [lcov](https://github.com/linux-test-project/lcov): grafical
      reports for line and function 
    - install with: 
    ```bash
    sudo apt update && sudo apt install lcov
    ```
    - or add to `default.nix`
