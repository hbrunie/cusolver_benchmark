#define DATATYPE double
#define SIGMA_I(i,n) ((DATATYPE)1. - (i-(DATATYPE)1.)*(n-(DATATYPE)1.)*(1-1/(1-cond)))

void generate_diagonal_matrix(DATATYPE ** D, int n, DATATYPE cond);
