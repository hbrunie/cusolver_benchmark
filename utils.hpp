#define DATATYPE double
#define SIGMA_I(i,n,cond) ((DATATYPE)1. - (i-(DATATYPE)1.)*(n-(DATATYPE)1.)*(1-1/(1-cond)))

void generate_diagonal_matrix(DATATYPE ** d_D, int n, DATATYPE cond);
void generate_orthogonal_matrix(DATATYPE ** d_Q, int n);
//void get_diagonal_from(DATATYPE * h_D, DATATYPE * d_D, int n);
