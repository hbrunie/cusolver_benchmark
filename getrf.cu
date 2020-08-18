#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

#include "common.h"

void check_pivot(bool pivot_on){
    if (pivot_on){
        printf("pivot is on : compute P*A = L*U \n");
    }else{
        printf("pivot is off: compute A = L*U (not numerically stable)\n");
    }
}

void printMatrix(int m, int n, const double*A, int lda, const char* name)
{
    for(int row = 0 ; row < m ; row++){
        for(int col = 0 ; col < n ; col++){
            double Areg = A[row + col*lda];
            printf("%s(%d,%d) = %f\n", name, row+1, col+1, Areg);
        }
    }
}

double * generate_rhs(const int size){
    double * m = (double*) malloc(sizeof(double)*size);
    return m;
}


double * generate_squared_matrix(const int size){
    double * m = (double*) malloc(sizeof(double)*size*size);
    return m;
}

int main(int argc, char*argv[])
{
    cusolverDnHandle_t cusolverH = NULL;
    cudaStream_t stream = NULL;

    cusolverStatus_t status = CUSOLVER_STATUS_SUCCESS;
    cudaError_t cudaStat1 = cudaSuccess;
    cudaError_t cudaStat2 = cudaSuccess;
    cudaError_t cudaStat3 = cudaSuccess;
    cudaError_t cudaStat4 = cudaSuccess;
    const int m = 3;
    const int lda = m;
    const int ldb = m;
    const int size = lda*m;
    double *B = generate_rhs(size);
    double X[m]; /* X = A\B */
    double LU[lda*m]; /* L and U */
    int Ipiv[m];      /* host copy of pivoting sequence */
    int info = 0;     /* host copy of error info */

    double *d_A = NULL; /* device copy of A */
    double *d_B = NULL; /* device copy of B */
    int *d_Ipiv = NULL; /* pivoting sequence */
    int *d_info = NULL; /* error info */
    int  lwork = 0;     /* size of workspace */
    double *d_work = NULL; /* device workspace for getrf */

    double *A;
    int matrix_dim = 0;
    const char *input_file = NULL;
    handle_arguments(argc, argv, &matrix_dim, input_file);
    //Creating matrix A
    func_ret_t ret;
    if (input_file) {
        printf("Reading matrix from file %s\n", input_file);
        ret = create_matrix_from_file(&A, input_file, &matrix_dim);
        if (ret != RET_SUCCESS) {
            A = NULL;
            fprintf(stderr, "error create matrix from file %s\n", input_file);
            exit(EXIT_FAILURE);
        }
    }
    else if (matrix_dim) {
        printf("Creating matrix internally size=%d\n", matrix_dim);
        ret = create_matrix(&A, matrix_dim);
        if (ret != RET_SUCCESS) {
            A = NULL;
            fprintf(stderr, "error create matrix internally size=%d\n", matrix_dim);
            exit(EXIT_FAILURE);
        }
    }
    else {
        printf("No input file specified!\n");
        exit(EXIT_FAILURE);
    }

    const bool pivot_on = true;
    check_pivot(pivot_on);


    printf("A = (matlab base-1)\n");
    printMatrix(m, m, A, lda, "A");
    printf("=====\n");

    printf("B = (matlab base-1)\n");
    printMatrix(m, 1, B, ldb, "B");
    printf("=====\n");

    /* step 1: create cusolver handle, bind a stream */
    status = cusolverDnCreate(&cusolverH);
    assert(CUSOLVER_STATUS_SUCCESS == status);

    cudaStat1 = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    assert(cudaSuccess == cudaStat1);

    status = cusolverDnSetStream(cusolverH, stream);
    assert(CUSOLVER_STATUS_SUCCESS == status);

    /* step 2: copy A to device */
    cudaStat1 = cudaMalloc ((void**)&d_A, sizeof(double) * size);
    cudaStat2 = cudaMalloc ((void**)&d_B, sizeof(double) * m);
    cudaStat4 = cudaMalloc ((void**)&d_info, sizeof(int));
    assert(cudaSuccess == cudaStat1);
    assert(cudaSuccess == cudaStat2);
    assert(cudaSuccess == cudaStat3);
    assert(cudaSuccess == cudaStat4);

    cudaStat1 = cudaMemcpy(d_A, A, sizeof(double)*size, cudaMemcpyHostToDevice);
    cudaStat2 = cudaMemcpy(d_B, B, sizeof(double)*m, cudaMemcpyHostToDevice);
    assert(cudaSuccess == cudaStat1);
    assert(cudaSuccess == cudaStat2);

    /* step 3: query working space of getrf */
    status = cusolverDnDgetrf_bufferSize(
            cusolverH,
            m,
            m,
            d_A,
            lda,
            &lwork);
    assert(CUSOLVER_STATUS_SUCCESS == status);

    cudaStat1 = cudaMalloc((void**)&d_work, sizeof(double)*lwork);
    assert(cudaSuccess == cudaStat1);

    /* step 4: LU factorization */
    if (!pivot_on){
        d_Ipiv = NULL;
    }else{
        cudaStat2 = cudaMalloc ((void**)&d_Ipiv, sizeof(int) * m);
        assert(cudaSuccess == cudaStat2);
    }

    status = cusolverDnDgetrf(
            cusolverH,
            m,
            m,
            d_A,
            lda,
            d_work,
            d_Ipiv,
            d_info);
    cudaStat1 = cudaDeviceSynchronize();
    assert(CUSOLVER_STATUS_SUCCESS == status);
    assert(cudaSuccess == cudaStat1);

    if (pivot_on)
        cudaStat1 = cudaMemcpy(Ipiv , d_Ipiv, sizeof(int)*m,
                cudaMemcpyDeviceToHost);

    cudaStat2 = cudaMemcpy(LU, d_A, sizeof(double)*lda*m,
            cudaMemcpyDeviceToHost);
    cudaStat3 = cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost);
    assert(cudaSuccess == cudaStat1);
    assert(cudaSuccess == cudaStat2);
    assert(cudaSuccess == cudaStat3);

    if ( 0 > info ){
        printf("%d-th parameter is wrong \n", -info);
        exit(1);
    }
    if (pivot_on){
        printf("pivoting sequence, matlab base-1\n");
        for(int j = 0 ; j < m ; j++){
            printf("Ipiv(%d) = %d\n", j+1, Ipiv[j]);
        }
    }
    printf("L and U = (matlab base-1)\n");
    printMatrix(m, m, LU, lda, "LU");
    printf("=====\n");

    status = cusolverDnDgetrs(
            cusolverH,
            CUBLAS_OP_N,
            m,
            1, /* nrhs */
            d_A,
            lda,
            d_Ipiv,
            d_B,
            ldb,
            d_info);

    cudaStat1 = cudaDeviceSynchronize();
    assert(CUSOLVER_STATUS_SUCCESS == status);
    assert(cudaSuccess == cudaStat1);

    cudaStat1 = cudaMemcpy(X , d_B, sizeof(double)*m, cudaMemcpyDeviceToHost);
    assert(cudaSuccess == cudaStat1);

    printf("X = (matlab base-1)\n");
    printMatrix(m, 1, X, ldb, "X");
    printf("=====\n");

    /* free resources */
    if (d_A    ) cudaFree(d_A);
    if (d_B    ) cudaFree(d_B);
    if (d_Ipiv ) cudaFree(d_Ipiv);
    if (d_info ) cudaFree(d_info);
    if (d_work ) cudaFree(d_work);

    if (cusolverH   ) cusolverDnDestroy(cusolverH);
    if (stream      ) cudaStreamDestroy(stream);

    cudaDeviceReset();

    return 0;
}

