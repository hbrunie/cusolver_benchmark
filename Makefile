all: tests

CXX=nvcc -g

%.o: %.cu
	$(CXX) -c $< -I.

%.o: %.cpp
	$(CXX) -c $< -I.

tests: main.o test_driven.o utils.o
	$(CXX) -o $@ $^ -lcurand


getrf: getrf.cu common.c
	nvcc -c $^ -std=c++14 -I. -g -lineno -I Common/
	g++ -fopenmp common.o getrf.o -o $@ -L${CUDA_ROOT}/lib64/ -lcusolver -lcudart -g

clean:
	rm -f getrf *.o
