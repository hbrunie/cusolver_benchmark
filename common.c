#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <getopt.h>

#include "common.h"

static struct option long_options[] = {
  /* name, has_arg, flag, val */
  {"input", 1, NULL, 'i'},
  {"size", 1, NULL, 's'},
  {"verify", 0, NULL, 'v'},
  {0,0,0,0}
};

void handle_arguments(int argc, char * argv[], int * matrix_dim, const char *input_file, int * rhs_matrix_dim, const char * rhs_input_file){
  int opt, option_index=0;

  while ((opt = getopt_long(argc, argv, "::vs:i:", 
                            long_options, &option_index)) != -1 ) {
    switch(opt){
    case 'i':
      input_file = optarg;
      break;
    case 'r':
      rhs_input_file = optarg;
      break;
    case 's':
      *matrix_dim = atoi(optarg);
      *rhs_matrix_dim = atoi(optarg);
      printf("Generate input matrix internally, size =%d\n", *matrix_dim);
      break;
    case '?':
      fprintf(stderr, "invalid option\n");
      break;
    case ':':
      fprintf(stderr, "missing argument\n");
      break;
    default:
      fprintf(stderr, "Usage: %s [-v] [-s matrix_size|-i input_file]\n",
	      argv[0]);
      exit(EXIT_FAILURE);
    }
  }
}

void stopwatch_start(stopwatch *sw){
    if (sw == NULL)
        return;

    bzero(&sw->begin, sizeof(struct timeval));
    bzero(&sw->end  , sizeof(struct timeval));

    gettimeofday(&sw->begin, NULL);
}

void stopwatch_stop(stopwatch *sw){
    if (sw == NULL)
        return;

    gettimeofday(&sw->end, NULL);
}

double 
get_interval_by_sec(stopwatch *sw){
    if (sw == NULL)
        return 0;
    return ((double)(sw->end.tv_sec-sw->begin.tv_sec)+(double)(sw->end.tv_usec-sw->begin.tv_usec)/1000000);
}

int 
get_interval_by_usec(stopwatch *sw){
    if (sw == NULL)
        return 0;
    return ((sw->end.tv_sec-sw->begin.tv_sec)*1000000+(sw->end.tv_usec-sw->begin.tv_usec));
}

func_ret_t 
create_rhs_matrix_from_file(DATATYPE **mp, const char* filename, int *size_p){
  int i, j, size;
  DATATYPE *m;
  FILE *fp = NULL;

  fp = fopen(filename, "rb");
  if ( fp == NULL) {
      return RET_FAILURE;
  }

  fscanf(fp, "%d\n", &size);

  m = (DATATYPE*) malloc(sizeof(DATATYPE)*size*1);
  if ( m == NULL) {
      fclose(fp);
      return RET_FAILURE;
  }
  for (i=0; i < size; i++) {
      fscanf(fp, "%f ", m+i);
  }

  fclose(fp);

  *size_p = size;
  *mp = m;

  return RET_SUCCESS;
}

func_ret_t 
create_matrix_from_file(DATATYPE **mp, const char* filename, int *size_p){
  int i, j, size;
  DATATYPE *m;
  FILE *fp = NULL;

  fp = fopen(filename, "rb");
  if ( fp == NULL) {
      return RET_FAILURE;
  }

  fscanf(fp, "%d\n", &size);

  m = (DATATYPE*) malloc(sizeof(DATATYPE)*size*size);
  if ( m == NULL) {
      fclose(fp);
      return RET_FAILURE;
  }

  for (i=0; i < size; i++) {
      for (j=0; j < size; j++) {
          fscanf(fp, "%f ", m+i*size+j);
      }
  }

  fclose(fp);

  *size_p = size;
  *mp = m;

  return RET_SUCCESS;
}

func_ret_t
create_rhs_matrix_from_random(DATATYPE **mp, int size){
  DATATYPE *m;

  m = (DATATYPE*) malloc(sizeof(DATATYPE)*size);
  srand(time(NULL));
  for (int i = 0; i < size; i++)
      m[i] = GET_RAND_FP;

  *mp = m;
  return RET_SUCCESS;
}

func_ret_t
create_matrix_from_random(DATATYPE **mp, int size){
  DATATYPE *l, *u, *m;
  int i,j,k;

  srand(time(NULL));

  m = (DATATYPE*)malloc(size*size*sizeof(DATATYPE));
  if ( m == NULL)
    return RET_FAILURE;

  l = (DATATYPE*)malloc(size*size*sizeof(DATATYPE));
  if ( l == NULL)
    return RET_FAILURE;

  u = (DATATYPE*)malloc(size*size*sizeof(DATATYPE));
  if ( u == NULL) {
      free(l);
      return RET_FAILURE;
  }

  for (i = 0; i < size; i++) {
      for (j=0; j < size; j++) {
          if (i>j) {
              l[i*size+j] = GET_RAND_FP;
          } else if (i == j) {
              l[i*size+j] = 1;
          } else {
              l[i*size+j] = 0;
          }
      }
  }

  for (j=0; j < size; j++) {
      for (i=0; i < size; i++) {
          if (i>j) {
              u[j*size+i] = 0;
          }else {
              u[j*size+i] = GET_RAND_FP; 
          }
      }
  }

  for (i=0; i < size; i++) {
      for (j=0; j < size; j++) {
          for (k=0; k <= MIN(i,j); k++)
            m[i*size+j] = l[i*size+k] * u[j*size+k];
      }
  }

  free(l);
  free(u);

  *mp = m;

  return RET_SUCCESS;
}

void
matrix_multiply(DATATYPE *inputa, DATATYPE *inputb, DATATYPE *output, int size){
  int i, j, k;

  for (i=0; i < size; i++)
    for (k=0; k < size; k++)
      for (j=0; j < size; j++)
        output[i*size+j] = inputa[i*size+k] * inputb[k*size+j];

}

func_ret_t
lud_verify(DATATYPE *m, DATATYPE *lu, int matrix_dim){
  int i,j,k;
  DATATYPE *tmp = (DATATYPE*)malloc(matrix_dim*matrix_dim*sizeof(DATATYPE));

  for (i=0; i < matrix_dim; i ++)
    for (j=0; j< matrix_dim; j++) {
        DATATYPE sum = 0;
        DATATYPE l,u;
        for (k=0; k <= MIN(i,j); k++){
            if ( i==k)
              l=1;
            else
              l=lu[i*matrix_dim+k];
            u=lu[k*matrix_dim+j];
            sum+=l*u;
        }
        tmp[i*matrix_dim+j] = sum;
    }
  /* printf(">>>>>LU<<<<<<<\n"); */
  /* for (i=0; i<matrix_dim; i++){ */
  /*   for (j=0; j<matrix_dim;j++){ */
  /*       printf("%f ", lu[i*matrix_dim+j]); */
  /*   } */
  /*   printf("\n"); */
  /* } */
  /* printf(">>>>>result<<<<<<<\n"); */
  /* for (i=0; i<matrix_dim; i++){ */
  /*   for (j=0; j<matrix_dim;j++){ */
  /*       printf("%f ", tmp[i*matrix_dim+j]); */
  /*   } */
  /*   printf("\n"); */
  /* } */
  /* printf(">>>>>input<<<<<<<\n"); */
  /* for (i=0; i<matrix_dim; i++){ */
  /*   for (j=0; j<matrix_dim;j++){ */
  /*       printf("%f ", m[i*matrix_dim+j]); */
  /*   } */
  /*   printf("\n"); */
  /* } */

  for (i=0; i<matrix_dim; i++){
      for (j=0; j<matrix_dim; j++){
          if ( fabs(m[i*matrix_dim+j]-tmp[i*matrix_dim+j]) > 0.0001)
            printf("dismatch at (%d, %d): (o)%f (n)%f\n", i, j, m[i*matrix_dim+j], tmp[i*matrix_dim+j]);
      }
  }
  free(tmp);
}

void
matrix_duplicate(DATATYPE *src, DATATYPE **dst, int matrix_dim) {
    int s = matrix_dim*matrix_dim*sizeof(DATATYPE);
   DATATYPE *p = (DATATYPE *) malloc (s);
   memcpy(p, src, s);
   *dst = p;
}

void
print_matrix(DATATYPE *m, int matrix_dim) {
    int i, j;
    for (i=0; i<matrix_dim;i++) {
      for (j=0; j<matrix_dim;j++)
        printf("%f ", m[i*matrix_dim+j]);
      printf("\n");
    }
}


// Generate well-conditioned matrix internally  by Ke Wang 2013/08/07 22:20:06
func_ret_t
create_well_conditionned_matrix(DATATYPE **mp, int size){
  DATATYPE *m;
  int i,j;
  DATATYPE lamda = -0.001;
  DATATYPE coe[2*size-1];
  DATATYPE coe_i =0.0;

  for (i=0; i < size; i++)
    {
      coe_i = 10*exp(lamda*i); 
      j=size-1+i;     
      coe[j]=coe_i;
      j=size-1-i;     
      coe[j]=coe_i;
    }

  m = (DATATYPE*) malloc(sizeof(DATATYPE)*size*size);
  if ( m == NULL) {
      return RET_FAILURE;
  }

  for (i=0; i < size; i++) {
      for (j=0; j < size; j++) {
	m[i*size+j]=coe[size-1-i+j];
      }
  }

  *mp = m;

  return RET_SUCCESS;
}
