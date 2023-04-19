AUTOPILOT_ROOT := /tools/software/xilinx/Vitis_HLS/2022.1
ASSEMBLE_SRC_ROOT := .

IFLAG += -I "${AUTOPILOT_ROOT}/include"
IFLAG += -I "${ASSEMBLE_SRC_ROOT}"
IFLAG += -I "/usr/include/x86_64-linux-gnu"

CFLAG += -fPIC -O3 -std=c++11 -mcmodel=large  -Wconversion
CC      = g++ 

CFLAG += -DCSIM_DEBUG # TODO Might have to remove this when building for syn

ALLOUT+= conv.o
ALLOUT+= sim

all: IFLAG +=  -g -DCSIM_DEBUG
all: $(ALLOUT) 

conv.o : conv.cpp util.h
	$(CC) $(CFLAG) -c $< -o $@

sim : sim.cpp util.h conv.h
	$(CC) $(CFLAG) -c $< -o $@

clean:
	clean:
	rm -f -r csim.d 
	rm -f *.out *.gcno *.gcda *.txt *.o *.d
