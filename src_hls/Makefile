AUTOPILOT_ROOT := /tools/software/xilinx/Vitis_HLS/2022.1
ASSEMBLE_SRC_ROOT := .

IFLAG += -I "${AUTOPILOT_ROOT}/include"
IFLAG += -I "${ASSEMBLE_SRC_ROOT}"
IFLAG += -I "/usr/include/x86_64-linux-gnu"

CFLAG += -fPIC -O3 -std=c++11 -mcmodel=large  -Wconversion
CC      = g++ 

ALLOUT+= conv.o
ALLOUT+= sim

all: IFLAG +=  -g -DCSIM_DEBUG
all: $(ALLOUT) 

conv.o : conv.cpp util.h
	$(CC) $(CFLAG) -DCSIM_DEBUG -c $< -o $@

csim : sim.cpp
	# We need to increase the stack size
	# This is our hack to not hit seg faults
	ulimit -s 32752 
	$(CC) $(CFLAG) -g -DCSIM_DEBUG $< -o $@

clean:
	rm -f -r csim.d 
	rm -f csim
	rm -f *.out *.gcno *.gcda *.txt *.o *.d

synth:
	vitis_hls script.tcl