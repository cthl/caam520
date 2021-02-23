#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "csr_matrix.h"

typedef struct
{
  int row;
  int col;
  double val;
} coo_t;

static int compare_coo(const void *left, const void *right)
{
  const coo_t *t0 = (coo_t*) left;
  const coo_t *t1 = (coo_t*) right;

  if (t0->row < t1->row) return -1;
  else if (t0->row > t1->row) return 1;
  else {
    if (t0->col < t1->col) return -1;
    else if (t0->col > t1->col) return 1;
  }

  return 0;
}

// Note: This function will only load *symmetric* matrices from matrix market
// files whose entries are sorted by the column index!
int csr_matrix_load(csr_matrix_t *matrix, const char *filename)
{
  char *ptr, *eptr;

  // Open matrix market file.
  FILE *fp = fopen(filename, "r");
  if (!fp) {
    return -1;
  }

  // Allocate buffer (for one line).
  const size_t buffer_size = 1024;
  char *buffer = (char*) malloc(buffer_size);
  if (!buffer) {
    fclose(fp);
    return -1;
  }

  // Read matrix dimensions and check if the matrix is symmetric.
  int symmetric = 0;
  while (1) {
    if (!fgets(buffer, buffer_size, fp)) {
      free(buffer);
      fclose(fp);
      return -1;
    }
    if (strstr(buffer, "%%") && strstr(buffer, "symmetric")) {
      symmetric = 1;
    }
    if (!strstr(buffer, "%")) {
      // Found the line that contains the matrix dimensions.
      break;
    }
  }
  const int num_rows = (int) strtol(buffer, &eptr, 10);
  if (num_rows == 0) {
    free(buffer);
    fclose(fp);
    return -1;
  }

  const int num_cols = (int) strtol(eptr, &eptr, 10);
  if (num_rows != num_cols) {
    free(buffer);
    fclose(fp);
    return -1;
  }

  int nnz = (int) strtol(eptr, NULL, 10);

  // Allocate COO array.
  coo_t *coo_array = (coo_t*) malloc(nnz*sizeof(coo_t));
  if (!coo_array) {
    free(buffer);
    fclose(fp);
    return -1;
  }

  // Read file line by line.
  int i = 0;
  while (1) {
    // Read next line.
    if (!fgets(buffer, buffer_size, fp)) {
      // Reached end of matrix market file.
      break;
    }

    // Get row, column, and (optional) value.
    ptr = buffer;
    const int row = strtol(ptr, &eptr, 10) - 1;
    if (ptr == eptr) {
      free(buffer);
      free(coo_array);
      fclose(fp);
      return -1;
    }

    ptr = eptr;
    const int col = strtol(ptr, &eptr, 10) - 1;
    if (ptr == eptr) {
      free(buffer);
      free(coo_array);
      fclose(fp);
      return -1;
    }

    ptr = eptr;
    double val = strtod(ptr, &eptr);
    if (ptr == eptr) {
      // Value was not specified, so it must be one as per matrix market format.
      val = 1.0;
    }
    else if (val == 0.0) {
      // We found a true zero entry, which we will not add to the matrix.
      continue;
    }

    coo_array[i].row = row;
    coo_array[i].col = col;
    coo_array[i].val = val;
    i++;
  }

  free(buffer);
  fclose(fp);

  // If the matrix is symmetric, we need to repeat all off-diagonal entries
  // with swapped row and column indices.
  if (symmetric) {
    int nnz_offdiag = 0;
    for (int i = 0; i < nnz; i++) {
      if (coo_array[i].row != coo_array[i].col) nnz_offdiag++;
    }

    coo_array = (coo_t*) realloc(coo_array, (nnz + nnz_offdiag)*sizeof(coo_t));

    int i_swapped = nnz;
    for (int i = 0; i < nnz; i++) {
      if (coo_array[i].row != coo_array[i].col) {
        coo_array[i_swapped].row = coo_array[i].col;
        coo_array[i_swapped].col = coo_array[i].row;
        coo_array[i_swapped].val = coo_array[i].val;
        i_swapped++;
      }
    }

    nnz += nnz_offdiag;
  }

  // Sort COO array.
  qsort(coo_array, nnz, sizeof(coo_t), compare_coo);

  // Convert from COO to CSR format.
  matrix->row_ptr = (int*) malloc((num_rows + 1)*sizeof(int));
  matrix->col_ind = (int*) malloc(nnz*sizeof(int));
  matrix->val = (double*) malloc(nnz*sizeof(double));
  if (!matrix->row_ptr || !matrix->col_ind || !matrix->val) {
    csr_matrix_free(matrix);
    free(buffer);
    fclose(fp);
    return -1;
  }

  i = 0;
  for (int row = 0; row < num_rows; row++) {
    matrix->row_ptr[row] = i;
    while (coo_array[i].row == row) {
      matrix->col_ind[i] = coo_array[i].col;
      matrix->val[i] = coo_array[i].val;
      i++;
    }
  }
  matrix->row_ptr[num_rows] = nnz;

  free(coo_array);
  return num_rows;
}

int csr_matrix_free(csr_matrix_t *matrix)
{
  free(matrix->row_ptr);
  free(matrix->col_ind);
  free(matrix->val);
  return 0;
}
