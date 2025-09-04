# Coverage Example
Standalone example for FU Coverage

## Contents:
.
└── coverage
    ├── Makefile
    └── src
        ├── relu_tb.cpp: verilator testbench
        └── Relu.v: example FU to cover

### Run example:
```bash
make coverage
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

## Optional graphical report:
- `cov-lcov` makefile target
    - requires [lcov](https://github.com/linux-test-project/lcov): grafical
      reports for line and function 
    - install with: 
    ```bash
    sudo apt update && sudo apt install lcov
    ```
    - or add to `default.nix`
