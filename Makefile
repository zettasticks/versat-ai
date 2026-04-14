# SPDX-FileCopyrightText: 2025 IObundle
#
# SPDX-License-Identifier: MIT

CORE := versat_ai

SIMULATOR ?= verilator
SYNTHESIZER ?= yosys
LINTER ?= spyglass
BOARD ?= iob_aes_ku040_db_g

TEST := Generated # Run setupTest.py to see the possible values for this

PYTHON_ENV := ../python_env
VERSAT_ACCEL := ./submodules/iob_versat/iob_versat.py
VERSAT_SUBMODULE := ./submodules/VERSAT
ALL_SCRIPTS := $(wildcard ./scripts/*.py)

VERSION ?=$(shell cat ./versat_ai.py | grep version | cut -d '"' -f 4)

BUILD_DIR ?= ../versat_ai_V$(VERSION)

INIT_MEM ?= 1
USE_EXTMEM ?= 1
USE_INTMEM ?= 0
USE_ETHERNET ?= 0
TESTER ?= 0
TESTER_SIM ?= 0

ifneq ($(DEBUG),)
EXTRA_ARGS +=--debug_level $(DEBUG)
endif

make-python-env: $(PYTHON_ENV)

./tests/alexnet/model.onnx: 
	./scripts/downloadAlexnet.sh

$(PYTHON_ENV):
	./scripts/makePythonEnv.sh

# Resources that are generated as they are needed. Mostly the generated tests and Versat
make-versat-accel: $(VERSAT_ACCEL)

$(VERSAT_ACCEL): versatSpec.txt
	@rm -f $(VERSAT_SUBMODULE)/iob_versat.py
	nix-shell --run "python3 ./scripts/versatGenerate.py"

generate-test:
	bash -c "source $(PYTHON_ENV)/bin/activate ; python3 ./setupTest.py $(TEST)"

test-setup: $(PYTHON_ENV) $(VERSAT_ACCEL) ./tests/alexnet/model.onnx generate-test
	mkdir -p hardware/simulation
	nix-shell --run "py2hwsw $(CORE) setup --no_verilog_lint --py_params 'use_intmem=$(USE_INTMEM):use_extmem=$(USE_EXTMEM):init_mem=$(INIT_MEM):use_ethernet=$(USE_ETHERNET):include_tester=$(TESTER):tester_sim=$(TESTER_SIM)' $(EXTRA_ARGS);"
	cp -r ./resources ../versat_ai_V$(VERSION)/
	cp -r submodules/iob_versat/software ../versat_ai_V$(VERSION)/ # Since python file was not being copied and we need a python script from inside software
	cp -r ./software ../versat_ai_V$(VERSION)/
	cp ./scripts/console.py ../versat_ai_V$(VERSION)/scripts
	-cp ./scripts/console.py ../versat_ai_V$(VERSION)/tester/scripts
	cp ./scripts/console_ethernet.py ../versat_ai_V$(VERSION)/scripts
	-cp ./scripts/console_ethernet.py ../versat_ai_V$(VERSION)/tester/scripts
	cp ./scripts/makehex.py ../versat_ai_V$(VERSION)/scripts
	-cp ./scripts/makehex.py ../versat_ai_V$(VERSION)/tester/scripts
	-cp ./software/makehex.c ../versat_ai_V$(VERSION)/tester/software
	
.PHONY: make-python-env make-versat-accel generate-test test-setup

pc-emul-run: test-setup
	nix-shell --run "make -C ../$(CORE)_V$(VERSION)/ pc-emul-run"

sim-run: test-setup
	nix-shell --run "make -C ../$(CORE)_V$(VERSION)/ sim-run SIMULATOR=$(SIMULATOR)"

tester-sim-run:
	make test-setup TESTER=1 TESTER_SIM=1
	nix-shell --run "make -C ../$(CORE)_V$(VERSION)/tester sim-run SIMULATOR=$(SIMULATOR)"

# For some reason the vivado build.tcl is being overwritten by py2. Need to copy it before 
fpga-run: test-setup
	nix-shell --run "make -C ../$(CORE)_V$(VERSION)/ fpga-sw-build BOARD=$(BOARD)"
	cp ./hardware/fpga/vivado/build.tcl ../$(CORE)_V$(VERSION)/hardware/fpga/vivado
	make -C ../$(CORE)_V$(VERSION)/ fpga-run BOARD=$(BOARD)

# For some reason the vivado build.tcl is being overwritten by py2. Need to copy it before 
tester-fpga-run:
	make test-setup TESTER=1
	nix-shell --run "make -C ../$(CORE)_V$(VERSION)/tester/ fpga-sw-build BOARD=$(BOARD)"
	cp ./hardware/fpga/vivado/build.tcl ../$(CORE)_V$(VERSION)/tester/hardware/fpga/vivado
	make -C ../$(CORE)_V$(VERSION)/tester/ fpga-run BOARD=$(BOARD)

fpga-build: test-setup
	nix-shell --run "make -C ../$(CORE)_V$(VERSION)/ fpga-sw-build BOARD=$(BOARD)"
	cp ./hardware/fpga/vivado/build.tcl ../$(CORE)_V$(VERSION)/hardware/fpga/vivado
	make -C ../$(CORE)_V$(VERSION)/ fpga-build BOARD=$(BOARD)

fpga-build-2: test-setup
	nix-shell --run "make -C ../$(CORE)_V*/tester/ fpga-sw-build BOARD=$(BOARD)" && make -C ../$(CORE)_V*/tester/ fpga-build BOARD=$(BOARD)
	
fpga-run-2: fpga-build-2
	make -C ../$(CORE)_V*/tester/ fpga-run BOARD=$(BOARD)

.PHONY: pc-emul-run sim-run tester-sim-run fpga-run tester-fpga-run fpga-build fpga-build-2 fpga-run-2

# Need to be inside nix-shell for fast rules to work. Mostly used to speed up development instead of waiting for setup everytime
fast-versat:
	python3 ./scripts/versatGenerate.py

fast-pc-no-generate:
	cp -r ./resources ../versat_ai_V$(VERSION)/
	cp -r software ../versat_ai_V$(VERSION)/
	cp -r submodules/iob_versat/software ../versat_ai_V$(VERSION)/
	make -C ../versat_ai_V$(VERSION)/ pc-emul-run

fast-pc-soft: fast-versat
	cp -r ./resources ../versat_ai_V$(VERSION)/
	cp -r software ../versat_ai_V$(VERSION)/
	cp -r submodules/iob_versat/software ../versat_ai_V$(VERSION)/
	make -C ../versat_ai_V$(VERSION)/ pc-emul-run

fast-pc-hard: fast-versat
	cp -r software ../versat_ai_V$(VERSION)/
	cp -r hardware ../versat_ai_V$(VERSION)/
	cp -r submodules/iob_versat/software ../versat_ai_V$(VERSION)/
	cp -r submodules/iob_versat/hardware ../versat_ai_V$(VERSION)/
	make -C ../versat_ai_V$(VERSION)/ pc-emul-run

fast-sim-run:
	cp -r resources ../versat_ai_V$(VERSION)/
	cp -r software ../versat_ai_V$(VERSION)/
	cp -r hardware ../versat_ai_V$(VERSION)/
	cp -r submodules/iob_versat/hardware ../versat_ai_V$(VERSION)/
	cp -r submodules/iob_versat/software ../versat_ai_V$(VERSION)/
	make -C ../versat_ai_V$(VERSION)/ sim-run SIMULATOR=$(SIMULATOR) VCD=$(VCD)

fast-tester:
	cp -r resources ../versat_ai_V$(VERSION)/
	cp -r software ../versat_ai_V$(VERSION)/
	cp -r submodules/iob_system_tester/software ../versat_ai_V$(VERSION)/tester
	make -C ../versat_ai_V$(VERSION)/tester sim-run SIMULATOR=$(SIMULATOR) VCD=$(VCD)

fast-only-sim-run:
	make -C ../versat_ai_V$(VERSION)/ sim-run SIMULATOR=$(SIMULATOR) VCD=$(VCD)

fast-fpga:
	cp -r resources ../versat_ai_V$(VERSION)/
	cp -r software ../versat_ai_V$(VERSION)/
	cp -r submodules/iob_versat/software ../versat_ai_V$(VERSION)/	
	make -C ../$(CORE)_V$(VERSION)/ fpga-sw-build BOARD=$(BOARD)
	make -C ../$(CORE)_V$(VERSION)/ fpga-run BOARD=$(BOARD)

fast-fpga-tester:
	cp -r resources ../versat_ai_V$(VERSION)/
	cp -r software ../versat_ai_V$(VERSION)/
	cp -r submodules/iob_system_tester/software ../versat_ai_V$(VERSION)/tester
	make -C ../$(CORE)_V$(VERSION)/tester fpga-sw-build BOARD=$(BOARD)
	make -C ../$(CORE)_V$(VERSION)/tester fpga-run BOARD=$(BOARD)

.PHONY: fast-versat fast-pc-no-generate fast-pc-soft fast-pc-hard fast-sim-run fast-tester fast-only-sim-run fast-fpga fast-fpga-tester

versat-generate:
	rm -f $(VERSAT_ACCEL)
	$(MAKE) $(VERSAT_ACCEL)

clean-test:
	@rm -rf ./resources

clean:
	nix-shell --run "py2hwsw $(CORE) clean --build_dir '$(BUILD_DIR)'"
	@rm -rf ./submodules/iob_versat
	@rm -rf ../*.summary ../*.rpt
	@rm -rf ./*.rpt
	@find . -name \*~ -delete

# Remove all __pycache__ folders with python bytecode
python-cache-clean:
	find . -name "*__pycache__" -exec rm -rf {} \; -prune

.PHONY: versat-generate  clean-test clean python-cache-clean

# Use --fu-dir to list all FUs for linting
VLINT_FLAGS += --fu-dir ./hardware/src
VLINT_FLAGS += --fu-dir ./hardware/units
VLINT_FLAGS += --fu-dir ./submodules/VERSAT/hardware/src/units
# Use build directory to find all verilog sources and headers
VLINT_FLAGS += -d ../versat_ai_V$(VERSION)/hardware/src
VLINT_FLAGS += -c ./hardware/lint
VLINT_FLAGS += -c ./submodules/VERSAT/hardware/lint
VLINT_FLAGS += -o lint.rpt
lint-all-fus: clean test-setup
	nix-shell --run "./scripts/verilog_linter.py $(VLINT_FLAGS)"
	cat lint.rpt

FU?=iob_fp_clz
lint-fu: clean test-setup
	nix-shell --run "./scripts/verilog_linter.py $(VLINT_FLAGS) --fu $(FU)"
	cat lint.rpt

coverage-all-fus: clean test-setup
	nix-shell --run "make -C ../versat_ai_V$(VERSION)/hardware/simulation/coverage all"

.PHONY: lint-all-fus lint-fu coverage-all-fus
