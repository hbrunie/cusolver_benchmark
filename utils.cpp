#include "utils.hpp"

/* Generate positive eignenvalues, singular values are arithmetically distributed
 * Sigma_i = 1 - (\frac{i - 1}{n-1})(1-\frac{1}{cond})
 * where cond is the condition number of the matrix with these eigenvalues.
 * Condition number is defined with the infinite norm as: cond = norm(A) \times norm(A^{-1})
 *
 * Matrix D heap memory management must be handle outside the function.
 * D is a square matrix of size n by n.
 */
void generate_diagonal_matrix(DATATYPE * D, int n, DATATYPE cond){
    // Reset to 0 values inside D
    memset(D,0,n*n*sizeof(DATATYPE));
    // Fill D with diagonal values
    for(int i=0; i<n; i++)
        D[n*i+i] = (DATATYPE)1. - (i-(DATATYPE)1.)(n-(DATATYPE)1.)*(1-1/(1-cond));
}

/* Matrix Q heap memory management must be handle outside the function.
 * Q is a square matrix of size n by n.
 */
void generate_orthogonal_matrix(DATATYPE * Q, int n){
    // Reset to 0 values inside Q
    memset(D,0,n*n*sizeof(DATATYPE));
    generate_random_matrix(Q,n);
    orthogonalize_matrix(Q,n)
}
