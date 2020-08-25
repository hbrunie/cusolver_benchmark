#include <cstdlib>
#include <cstring>
#include <cassert>
#include <cuda_runtime.h>
#include "cublas_v2.h"

#include "utils.hpp"
#include <iostream>
using namespace std;

void test_diagonal_matrix(){
    DATATYPE * d_D;
    int n = 10;
    DATATYPE * h_D = (DATATYPE*) malloc(sizeof(DATATYPE) * n*n);
    DATATYPE condarray[6] = {1e3,2e3,5e3,1e4,2e4,5e4};
    cudaError_t cudaStat1 = cudaSuccess;
    // tested code
    for (int condInd = 0; condInd<6; condInd++){
        double cond = condarray[condInd];
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
    free(d_D);
    cudaFree(d_D);
}

void test_orthogonal_matrix(){
    double * d_Q;
    int n = 10;
    double * h_Q = (double*) malloc(sizeof(double) * n*n);
    double cond = 1000.;
    cudaError_t cudaStat1 = cudaSuccess;
    //tested code
    generate_orthogonal_matrix(&d_Q, n,cond);

    cudaStat1 = cudaMemcpy(h_Q, d_Q, sizeof(DATATYPE)*n*n, cudaMemcpyDeviceToHost);
    assert(cudaSuccess == cudaStat1);

// step 6: measure R = I - Q**T*Q
    DATATYPE * h_R = (DATATYPE*) malloc(sizeof(DATATYPE)*n*n);
    DATATYPE * d_R;
    cudaStat1 = cudaMalloc((void**)&(*d_R), sizeof(DATATYPE)*n*n);
    assert(cudaSuccess == cudaStat1);
    memset(h_R, 0, sizeof(DATATYPE)*n*n);
    for(int j = 0 ; j < n ; j++)
        h_R[j + n*j] = 1.0; // R(j,j)=1

    cudaStat1 = cudaMemcpy(d_R, h_R, sizeof(DATATYPE)*n*n, cudaMemcpyHostToDevice);
    assert(cudaSuccess == cudaStat1);

    // R = -Q**T*Q + I
    cublasHandle_t cublasH;
    cublasStatus_t cublas_status = cublasCreate(&cublasH);
    const DATATYPE h_minus_one = -1.;
    const DATATYPE h_one = 1.;
    assert(CUBLAS_STATUS_SUCCESS == cublas_status);
    cublas_status = cublasDgemm_v2(
        cublasH,
        CUBLAS_OP_T, // Q**T
        CUBLAS_OP_N, // Q
        n, // number of rows of R
        n, // number of columns of R
        n, // number of columns of Q**T
        &h_minus_one, /* host pointer */
        d_Q, // Q**T
        n,
        d_Q, // Q
        n,
        &h_one, /* hostpointer */
        d_R,
        n);
    assert(CUBLAS_STATUS_SUCCESS == cublas_status);

    DATATYPE dR_nrm2 = 0.0;
    cublas_status = cublasDnrm2_v2(
        cublasH, n*n, d_R, 1, &dR_nrm2);
    assert(CUBLAS_STATUS_SUCCESS == cublas_status);

    printf("|I - Q**T*Q| = %E\n", dR_nrm2);

}
