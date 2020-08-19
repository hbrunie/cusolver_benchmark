#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

#include "helper_cuda.h"
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

int main(int argc, char*argv[])
{
    cusolverDnHandle_t cusolverH = NULL;
    cudaStream_t stream = NULL;

    cusolverStatus_t status = CUSOLVER_STATUS_SUCCESS;
    cudaError_t cudaStat1 = cudaSuccess;
    cudaError_t cudaStat2 = cudaSuccess;
    cudaError_t cudaStat3 = cudaSuccess;
    cudaError_t cudaStat4 = cudaSuccess;
    int info = 0;     /* host copy of error info */

    double *d_A = NULL; /* device copy of A */
    double *d_B = NULL; /* device copy of B */
    int *d_Ipiv = NULL; /* pivoting sequence */
    int *d_info = NULL; /* error info */
    int  lwork = 0;     /* size of workspace */
    double *d_work = NULL; /* device workspace for getrf */

    double *A,*B;
    int matrix_dim = 0;
    int rhs_matrix_dim = 0;
    const char *input_file = NULL;
    const char *rhs_input_file = NULL;
    handle_arguments(argc, argv, &matrix_dim, input_file, &rhs_matrix_dim, rhs_input_file);
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
        ret = create_matrix_from_random(&A, matrix_dim);
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

    if (rhs_input_file) {
        ret = create_rhs_matrix_from_file(&B, rhs_input_file, &rhs_matrix_dim);
        if (ret != RET_SUCCESS) {
            B = NULL;
            fprintf(stderr, "error create matrix from file %s\n", rhs_input_file);
            exit(EXIT_FAILURE);
        }
    }
    else if (rhs_matrix_dim) {
        printf("Creating RHS matrix internally size=%d\n", rhs_matrix_dim);
        ret = create_rhs_matrix_from_random(&B, rhs_matrix_dim);
        if (ret != RET_SUCCESS) {
            B = NULL;
            fprintf(stderr, "error create matrix internally size=%d\n", rhs_matrix_dim);
            exit(EXIT_FAILURE);
        }
    }
    else {
        printf("No input file specified!\n");
        exit(EXIT_FAILURE);
    }

    const int size = matrix_dim*matrix_dim;
    const int lda = matrix_dim;
    const int ldb = matrix_dim;
    double X[matrix_dim]; /* X = A\B */
    double LU[lda*matrix_dim]; /* L and U */
    int Ipiv[matrix_dim];      /* host copy of pivoting sequence */

    const bool pivot_on = true;
    float getrf_msecTotal = 0.0f, getrs_msecTotal = 0.0f;
    check_pivot(pivot_on);


    printf("A = (matlab base-1)\n");
    printMatrix(matrix_dim, matrix_dim, A, lda, "A");
    printf("=====\n");

    printf("B = (matlab base-1)\n");
    printMatrix(matrix_dim, 1, B, ldb, "B");
    printf("=====\n");

    /* step 1: create cusolver handle, bind a stream */
    status = cusolverDnCreate(&cusolverH);
    assert(CUSOLVER_STATUS_SUCCESS == status);

    cudaStat1 = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    assert(cudaSuccess == cudaStat1);

    status = cusolverDnSetStream(cusolverH, stream);
    assert(CUSOLVER_STATUS_SUCCESS == status);

    /* step 2: copy A to device */
    cudaStat1 = cudaMalloc ((void**)&d_A, sizeof(double) * matrix_dim *matrix_dim );
    cudaStat2 = cudaMalloc ((void**)&d_B, sizeof(double) * matrix_dim);
    cudaStat4 = cudaMalloc ((void**)&d_info, sizeof(int));
    assert(cudaSuccess == cudaStat1);
    assert(cudaSuccess == cudaStat2);
    assert(cudaSuccess == cudaStat3);
    assert(cudaSuccess == cudaStat4);

    cudaStat1 = cudaMemcpy(d_A, A, sizeof(double)*size, cudaMemcpyHostToDevice);
    cudaStat2 = cudaMemcpy(d_B, B, sizeof(double)*matrix_dim, cudaMemcpyHostToDevice);
    assert(cudaSuccess == cudaStat1);
    assert(cudaSuccess == cudaStat2);

    /* step 3: query working space of getrf */
    status = cusolverDnDgetrf_bufferSize(
            cusolverH,
            matrix_dim,
            matrix_dim,
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
        cudaStat2 = cudaMalloc ((void**)&d_Ipiv, sizeof(int) * matrix_dim);
        assert(cudaSuccess == cudaStat2);
    }

    cublasHandle_t handle;
    // Allocate CUDA events that we'll use for timing
    cublasCreate(&handle);
    cudaEvent_t start;
    checkCudaErrors(cudaEventCreate(&start));
    cudaEvent_t stop;
    checkCudaErrors(cudaEventCreate(&stop));

    cudaDeviceSynchronize();
    // Record the start event
    checkCudaErrors(cudaEventRecord(start, NULL));
    status = cusolverDnDgetrf(
            cusolverH,
            matrix_dim,
            matrix_dim,
            d_A,
            lda,
            d_work,
            d_Ipiv,
            d_info);
    cudaStat1 = cudaDeviceSynchronize();
    // Record the stop event
    checkCudaErrors(cudaEventRecord(stop, NULL));
    // Wait for the stop event to complete
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&getrf_msecTotal, start, stop));

    assert(CUSOLVER_STATUS_SUCCESS == status);
    assert(cudaSuccess == cudaStat1);

    if (pivot_on)
        cudaStat1 = cudaMemcpy(Ipiv , d_Ipiv, sizeof(int)*matrix_dim,
                cudaMemcpyDeviceToHost);

    cudaStat2 = cudaMemcpy(LU, d_A, sizeof(double)*lda*matrix_dim,
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
        for(int j = 0 ; j < matrix_dim ; j++){
            printf("Ipiv(%d) = %d\n", j+1, Ipiv[j]);
        }
    }
    printf("L and U = (matlab base-1)\n");
    printMatrix(matrix_dim, matrix_dim, LU, lda, "LU");
    printf("=====\n");

    cudaDeviceSynchronize();
    checkCudaErrors(cudaEventRecord(start, NULL));
    status = cusolverDnDgetrs(
            cusolverH,
            CUBLAS_OP_N,
            matrix_dim,
            1, /* nrhs */
            d_A,
            lda,
            d_Ipiv,
            d_B,
            ldb,
            d_info);

    cudaStat1 = cudaDeviceSynchronize();
    checkCudaErrors(cudaEventRecord(stop, NULL));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&getrs_msecTotal, start, stop));
    assert(CUSOLVER_STATUS_SUCCESS == status);
    assert(cudaSuccess == cudaStat1);

    cudaStat1 = cudaMemcpy(X , d_B, sizeof(double)*matrix_dim, cudaMemcpyDeviceToHost);
    assert(cudaSuccess == cudaStat1);

    printf("X = (matlab base-1)\n");
    printMatrix(matrix_dim, 1, X, ldb, "X");
    printf("=====\n");

    /* free resources */
    if (d_A    ) cudaFree(d_A);
    if (d_B    ) cudaFree(d_B);
    if (d_Ipiv ) cudaFree(d_Ipiv);
    if (d_info ) cudaFree(d_info);
    if (d_work ) cudaFree(d_work);

    if (cusolverH   ) cusolverDnDestroy(cusolverH);
    if (stream      ) cudaStreamDestroy(stream);

    cublasDestroy(handle);
    cudaDeviceReset();

    return 0;
}

