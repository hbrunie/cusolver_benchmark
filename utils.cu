#include <cstdio>
#include <curand.h>
#include <cuda_runtime.h>
#include <cassert>
#include "utils.hpp"

/* Generate positive eignenvalues, singular values are arithmetically distributed
 * Sigma_i = 1 - (\frac{i - 1}{n-1})(1-\frac{1}{cond})
 * where cond is the condition number of the matrix with these eigenvalues.
 * Condition number is defined with the infinite norm as: cond = norm(A) \times norm(A^{-1})
 *
 * Matrix D heap memory management must be handle outside the function.
 * D is a square matrix of size n by n.
 */
__global__
void
fill_diagonal_matrix(DATATYPE * d_D, int n, DATATYPE cond){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx%(n+1) == 0)
        d_D[idx] = SIGMA_I(idx%n,n,cond);
    else
        d_D[idx] = 0;
}

void generate_diagonal_matrix(DATATYPE ** d_D, int n, DATATYPE cond){
    cudaError_t cudaStat1 = cudaSuccess;
    cudaStat1 = cudaMalloc((void**)&(*d_D), sizeof(DATATYPE)*n*n);
    assert(cudaSuccess == cudaStat1);
    // Fill D with diagonal values
    fill_diagonal_matrix<<<1,n*n>>>(*d_D, n, cond);
}


/* Generate matrix */
void fill_random_matrix(double *A, int N){
    curandGenerator_t prng;
    curandCreateGenerator (&prng, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long) clock());
    curandGenerateUniformDouble(prng, A, N);
}

void orthogonalize_matrix(DATATYPE * d_A, int n){
}

/* Matrix Q heap memory management must be handle outside the function.
 * Q is a square matrix of size n by n.
 */
void generate_orthogonal_matrix(DATATYPE ** d_Q, int n, DATATYPE cond){
    cudaError_t cudaStat1 = cudaSuccess;
    cudaStat1 = cudaMalloc((void**)&(*d_Q), sizeof(DATATYPE)*n*n);
    assert(cudaSuccess == cudaStat1);
    // Reset to 0 values inside Q
    fill_random_matrix(*d_Q,n);
    orthogonalize_matrix(*d_Q,n);
}

void ConstantInitDouble(double *data, int size, double val) {
    for (int i = 0; i < size; ++i) {
        data[i] = val+i;
    }
}
