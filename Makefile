################################################################################
#
## Build script for projct
#
#################################################################################
CC=g++

CU_INC=-I/usr/local/cuda-9.2/include
CUDA_LIB =-L/usr/local/cuda-9.2/lib64 -lcudart 

CFLAGS= -Wall -gdwarf-2 -O3 -funroll-loops -fopenmp 
	CUFLAGS = -gencode arch=compute_61,code=sm_61
	CFILES = $(wildcard *.cpp)
	CUFILES = $(wildcard *.cu)
	OBJECTS = $(CFILES:.cpp=.o)
CU_OBJECTS = $(CUFILES:.cu=.o)

all : $(OBJECTS) $(CU_OBJECTS)
	$(CC) -m64 $^ $(CFLAGS) $(CU_INC) $(CUDA_LIB)  -o vBF_GPU.out

$(OBJECTS) : $(CFILES) *.h
	$(CC) -m64 $(CFILES) $(CFLAGS) $(CU_INC) $(CUDA_LIB) -c

$(CU_OBJECTS) : $(CUFILES) *.h
	nvcc -c $(CUFILES) $(CUFLAGS)

clean :
	rm *.o


