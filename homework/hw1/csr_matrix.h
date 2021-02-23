#ifndef CSR_MATRIX_H
#define CSR_MATRIX_H

typedef struct
{
  int *row_ptr;
  int *col_ind;
  double *val;
} csr_matrix_t;

int csr_matrix_load(csr_matrix_t *matrix, const char *filename);

int csr_matrix_free(csr_matrix_t *matrix);

#endif // CSR_MATRIX_H
