#
# COVERAGE
#

VCD?=0
SRC_DIR=src
VSRC=$(wildcard ./$(SRC_DIR)/*.v)
VERILATED_PATH=./obj_dir
SIM_EXE=./$(VERILATED_PATH)/Vrelu_tb

COV_FLAGS= -Wall -cc
# alias for --main --exe --build --timing
# --main: Generate a main() function to run the simulation.
# --exe: Generate an executable simulation program.
# --build: Compile and link the generated C++ code into an executable.
# --timing: Support delays, etc
COV_FLAGS+= --binary
COV_FLAGS+= --trace
COV_FLAGS+= --coverage
ifeq ($(VCD),1)
COV_FLAGS+=--trace -DVCD
endif
COV_FLAGS+= --top-module relu_tb
COV_RPT=custom_coverage.rpt

# Custom coverage file
SIM_EXE+=+verilator+coverage+file+$(COV_RPT)

cov-verilog-only merged.dat: clean
	# verilate design and testbench
	verilator $(COV_FLAGS) $(VSRC)
	# run simulation
	$(SIM_EXE)
	# merge coverage
	verilator_coverage --write merged.dat $(COV_RPT)
	# annotate coverage
	# --annotate <dir>: create coverage annotations in <dir>.
	#  					Note: only shows modules with missing coverage
	# --annotate-min <count>: set minimum threshold for toggle for sufficient coverage
	# --annotate-all: write annotations for all modules, even if 100% covered
	# more info: https://verilator.org/guide/latest/exe_verilator_coverage.html
	verilator_coverage --annotate cov_annotated --annotate-min 2 --annotate-all merged.dat
	./scripts/coverage_analyze.py ./cov_annotated -o $(COV_RPT)
	cat $(COV_RPT)

gen-lcov-report: merged.dat
	# generate lcov-style report (optional lcov dependency)
	# sudo apt install lcov
	# NOTE: only shows line and function coverage, no toggle coverage
	verilator_coverage --write-info coverage.info merged.dat
	genhtml coverage.info -o coverage_html

clean cov-clean:
	rm -rf $(VERILATED_PATH) uut.vcd
	rm -rf $(COV_LOGS) $(COV_RPT) merged.dat cov_annotated \
		coverage.info coverage_html
	rm -rf $(COV_RPT)
