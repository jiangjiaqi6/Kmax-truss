CXX=g++
CXXFLAGS=-O3 -std=c++11 -fopenmp -Iinclude 
NVCC=/usr/local/cuda/bin/nvcc
NVCCFLAGS= -O3 -std=c++11 -Iinclude 

all : truss_gpu

truss_gpu : main.o truss_count_gpu.o log.o gpu.o
	$(CXX) $(CXXFLAGS) -o truss_gpu main.o truss_count_gpu.o log.o gpu.o -L /usr/local/cuda/lib64 -lcudart

main.o : main.cpp
	$(CXX) $(CXXFLAGS) -c main.cpp -o main.o

truss_count_gpu.o : truss_count_gpu.cu
	$(NVCC) $(NVCCFLAGS) -dc truss_count_gpu.cu -o truss_count_gpu.o

gpu.o : truss_count_gpu.o
	$(NVCC) $(NVCCFLAGS) -dlink truss_count_gpu.o -o gpu.o

log.o : log.cpp
	$(CXX) $(CXXFLAGS) -c log.cpp -o log.o

.PHONY : clean
clean :
	rm truss_gpu *.o
