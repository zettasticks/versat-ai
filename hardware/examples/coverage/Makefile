#
# COVERAGE
#
SRC_DIR ?= src

VSRC?= $(wildcard ./$(SRC_DIR)/*.v)
VSRC+= $(wildcard ./$(SRC_DIR)/*.cpp)

VTOP ?=Relu
SIM_OBJ=V$(VTOP)

VFLAGS+= --coverage
VFLAGS+=--cc --exe --top-module $(VTOP)
VFLAGS+=-Wno-lint --Wno-UNOPTFLAT
VFLAGS+=--no-timing
ifeq ($(VCD),1)
VFLAGS+=--trace -DVCD
endif
VERILATED_PATH ?= ./obj_dir
COV_LOGS ?= ./logs
COV_RESULTS ?= merged.dat

COV_LCOV ?= coverage.info
COV_HTML ?= coverage_html

sim:
	# verilate design and testbench
	verilator $(VFLAGS) $(VSRC)
	cd $(VERILATED_PATH) && make -f $(SIM_OBJ).mk
	# run simulation
	$(VERILATED_PATH)/$(SIM_OBJ)

$(COV_RESULTS) coverage: clean sim
	# merge coverage databases
	verilator_coverage --write $(COV_RESULTS) $(wildcard $(COV_LOGS)/*.dat)
	# annotate coverage
	# --annotate <dir>: create coverage annotations in <dir>.
	#  					Note: only shows modules with missing coverage
	# --annotate-min <count>: set minimum threshold for toggle for sufficient coverage
	# --annotate-all: write annotations for all modules, even if 100% covered
	# more info: https://verilator.org/guide/latest/exe_verilator_coverage.html
	verilator_coverage --annotate cov_annotated --annotate-min 2 --annotate-all $(COV_RESULTS)

cov-lcov: $(COV_RESULTS)
	# generate lcov-style report (optional lcov dependency)
	# sudo apt install lcov
	# NOTE: only shows line and function coverage, no toggle coverage
	verilator_coverage --write-info $(COV_LCOV) $(COV_RESULTS)
	genhtml $(COV_LCOV) -o $(COV_HTML)
	xdg-open $(COV_HTML)/index.html

clean:
	rm -rf $(VERILATED_PATH)
	rm -rf $(COV_LOGS) coverage.dat $(COV_RESULTS) cov_annotated
	rm -rf $(COV_LCOV) $(COV_HTML)
