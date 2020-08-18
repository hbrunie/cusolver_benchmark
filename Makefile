all: getrf

getrf: getrf.cu common.c
	nvcc -c $^ -std=c++14 -I. -g -lineno
	g++ -fopenmp common.o getrf.o -o $@ -L${CUDA_ROOT}/lib64/ -lcusolver -lcudart -g

clean:
	rm -f getrf *.o
