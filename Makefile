all: getrf

getrf: getrf.cu common.c
	nvcc -c $^ -std=c++14 -I.
	g++ -fopenmp getrf.o -o $@ -L${CUDA_ROOT}/lib64/ -lcusolver -lcudart

clean:
	rm -f getrf *.o
