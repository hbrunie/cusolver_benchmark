#ifndef _COMMON_H
#define _COMMON_H

#include <time.h>
#include <sys/time.h>

#ifdef __cplusplus
extern "C" {
#endif

#define DATATYPE double

#define GET_RAND_FP ( (DATATYPE)rand() /   \
                     ((DATATYPE)(RAND_MAX)+(DATATYPE)(1)) )

#define MIN(i,j) ((i)<(j) ? (i) : (j))

typedef enum _FUNC_RETURN_CODE {
    RET_SUCCESS,
    RET_FAILURE
}func_ret_t;

typedef struct __stopwatch_t{
    struct timeval begin;
    struct timeval end;
}stopwatch;

void 
stopwatch_start(stopwatch *sw);

void 
stopwatch_stop (stopwatch *sw);

double 
get_interval_by_sec(stopwatch *sw);

int 
get_interval_by_usec(stopwatch *sw);

func_ret_t
create_matrix_from_file(DATATYPE **mp, const char *filename, int *size_p);

func_ret_t
create_matrix_from_random(DATATYPE **mp, int size);

func_ret_t
create_matrix(DATATYPE **mp, int size);

func_ret_t
lud_verify(DATATYPE *m, DATATYPE *lu, int size);

void
matrix_multiply(DATATYPE *inputa, DATATYPE *inputb, DATATYPE *output, int size);

void
matrix_duplicate(DATATYPE *src, DATATYPE **dst, int matrix_dim);

void
print_matrix(DATATYPE *mm, int matrix_dim);

void
handle_arguments(int argc, char * argv[], int * matrix_dim, const char *input_file);

#ifdef __cplusplus
}
#endif

#endif
