#include <cstdlib>
#include <cuda_runtime.h>

#include "utils.hpp"
#include <iostream>
using namespace std;

void test_diagonal_matrix(){
    double * d_D;
    int n = 10;
    double * h_D = (double*) malloc(sizeof(double) * n*n);
    double cond = 1000.;
    cudaError_t cudaStat1 = cudaSuccess;
    // tested code
    generate_diagonal_matrix(&d_D, n, cond);
    // testing code
    cudaStat1 = cudaMemcpy(h_D, d_D, sizeof(DATATYPE)*n*n, cudaMemcpyDeviceToHost);
    for(int i=0; i<n; i++){
        for(int j=0; j<n; j++){
            double tmp = SIGMA_I(i,n,cond);
            double current = h_D[i*n+j];
            if((i==j && current!=tmp)
            || (i!=j && current!= 0.)){
                exit(-1);
            }
        }
    }
}
