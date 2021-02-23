#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#include "csr_matrix.h"

// Use the Jacobi method to solve the linear system Ax = b to the desired
// tolerance, i.e., such that ||b - Ax||/||b|| < tol.
void jacobi(const csr_matrix_t *A,  // Sparse matrix
            double *x,              // Solution vector and initial guess
            const double *b,        // Right-hand side
            int n,                  // System size
            double omega,           // Relaxation coefficient
            double tol,             // Relative tolerance
            double *residual,       // Achieved relative residual
            int *iter)              // Number of iterations until convergence
{
  //////////////////////////////////////////////////////////////////////////////
  //                         Add your code here!                              //
  //              (Feel free to add more functions as needed.)                //
  //////////////////////////////////////////////////////////////////////////////
}

// Do not modify the main function!
int main(int argc, char **argv)
{
  if (argc != 4) {
    fprintf(stderr, "Usage: ./jacobi matrix_file tol omega\n");
    return -1;
  }

  // Read tolerance from command line.
  const double tol = atof(argv[2]);

  // Read relaxation coefficient from command line.
  const double omega = atof(argv[3]);

  // Read matrix from matrix market file.
  csr_matrix_t A;
  const int n = csr_matrix_load(&A, argv[1]);
  if (n < 1) {
    fprintf(stderr, "Error: Could not load matrix market file %s!\n", argv[1]);
    return -1;
  }

  // Create solution vector and right-hand side.
  double *x = (double*) malloc(n*sizeof(double));
  double *b = (double*) malloc(n*sizeof(double));
  if (!x || !b) {
    fprintf(stderr, "Error: Could not create vectors x and b!\n");
    free(x);
    csr_matrix_free(&A);
    return -1;
  }
  for (int i = 0; i < n; i++) {
    x[i] = 0.0;
    b[i] = 1.0;
  }

  // Call Jacobi method.
  int iter;
  double residual;
  jacobi(&A, x, b, n, omega, tol, &residual, &iter);

  // Report results.
  printf("Jacobi method converged in %d iter. to a rel. residual of %e.\n",
         iter,
         residual);
  printf("(Matrix: %s, rel. tol.: %e)\n", argv[1], tol);

  free(x);
  free(b);
  csr_matrix_free(&A);
  return 0;
}
