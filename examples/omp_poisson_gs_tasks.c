// omp_poisson_gs_tasks.c - Solve Poisson's equation using finite differences
//                          and the Gauss-Seidel method.

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <omp.h>

#define M_PI 3.141592653589793
#define IJK2INDEX(I, J, K, N) ((N)*(N)*(K) + (N)*(J) + (I))

double u_exact(double x, double y, double z)
{
  return sin(M_PI*x)*sin(M_PI*y)*sin(M_PI*z);
}

double f_rhs(double x, double y, double z)
{
  return 3.0*M_PI*M_PI*u_exact(x, y, z);
}

void gauss_seidel_block(double *u,
                        const double *f,
                        int n,
                        int k,
                        int j_start,
                        int n_thread)
{
  const double h = 1.0/(n + 1);
  const double h2 = h*h;

  for (int j = j_start; j < j_start + n_thread; j++) {
    for (int i = 0; i < n; i++) {
      double u_new = h2*f[IJK2INDEX(i, j, k, n)];

      if (i > 0)     u_new += u[IJK2INDEX(i - 1, j,     k,     n)];
      if (i < n - 1) u_new += u[IJK2INDEX(i + 1, j,     k,     n)];
      if (j > 0)     u_new += u[IJK2INDEX(i,     j - 1, k,     n)];
      if (j < n - 1) u_new += u[IJK2INDEX(i,     j + 1, k,     n)];
      if (k > 0)     u_new += u[IJK2INDEX(i,     j,     k - 1, n)];
      if (k < n - 1) u_new += u[IJK2INDEX(i,     j,     k + 1, n)];

      u_new /= 6.0;

      u[IJK2INDEX(i, j, k, n)] = u_new;
    } // i
  } // j
}

void gauss_seidel(double *u, const double *f, int n, int num_iter)
{
  #pragma omp parallel
  {
    #pragma omp single
    {
      const int num_threads = omp_get_num_threads();
      if (n%num_threads != 0) {
        fprintf(stderr, "Error: Number of threads must divide the domain width!\n");
        exit(-1);
      }
      const int n_thread = n/num_threads;

      for (int iter = 0; iter < num_iter; iter++) {
        for (int k = 0; k < n; k++) {
          for (int j_start = 0; j_start + n_thread <= n; j_start += n_thread) {
            const int self = IJK2INDEX(0, j_start, k, n);
            const int left = IJK2INDEX(0, j_start - n_thread, k, n);
            const int below = IJK2INDEX(0, j_start, k - 1, n);

            const int block_size = n*n_thread;

            if (k == 0 && j_start == 0) {
              #pragma omp task depend(out:u[self:block_size])
              gauss_seidel_block(u, f, n, k, j_start, n_thread);
            }
            else if (k == 0) {
              #pragma omp task depend(in:u[left:block_size]) \
                               depend(out:u[self:block_size])
              gauss_seidel_block(u, f, n, k, j_start, n_thread);
            }
            else if (j_start == 0) {
              #pragma omp task depend(in:u[below:block_size]) \
                               depend(out:u[self:block_size])
              gauss_seidel_block(u, f, n, k, j_start, n_thread);
            }
            else {
              #pragma omp task depend(in:u[below:block_size]) \
                               depend(in:u[left:block_size]) \
                               depend(out:u[self:block_size])
              gauss_seidel_block(u, f, n, k, j_start, n_thread);
            }
          } // j_start
        } // k
        // Wait for the entire Gauss-Seidel iteration to finish.
        # pragma omp taskwait
      } // iter
    }
  }
}

int main(int argc, char **argv)
{
  if (argc != 3) {
    fprintf(stderr, "Usage: ./omp_poisson_gs_tasks n num_iter\n");
    return -1;
  }

  // Get command line arguments.
  const int n = atoi(argv[1]);
  const int num_iter = atoi(argv[2]);

  const double h = 1.0/(n + 1);

  double *u = malloc(n*n*n*sizeof(double));
  double *f = malloc(n*n*n*sizeof(double));
  if (!u || !f) {
    fprintf(stderr, "Error: Could not allocate memory!\n");
    free(u);
    return -1;
  }

  // Set initial guess and right-hand side.
  for (int k = 0; k < n; k++) {
    for (int j = 0; j < n; j++) {
      for (int i = 0; i < n; i++) {
        const double x = (i + 1)*h;
        const double y = (j + 1)*h;
        const double z = (k + 1)*h;
        f[IJK2INDEX(i, j, k, n)] = f_rhs(x, y, z);
        u[IJK2INDEX(i, j, k, n)] = 0.0;
      }
    }
  }

  // Call Gauss-Seidel method.
  gauss_seidel(u, f, n, num_iter);

  // Compute error norm.
  double norm_error = 0.0;
  double norm_u_exact = 0.0;
  for (int k = 0; k < n; k++) {
    for (int j = 0; j < n; j++) {
      for (int i = 0; i < n; i++) {
        const double x = (i + 1)*h;
        const double y = (j + 1)*h;
        const double z = (k + 1)*h;
        const double u_exact_ijk = u_exact(x, y, z);

        norm_u_exact += u_exact_ijk*u_exact_ijk;

        const double diff = u[IJK2INDEX(i, j, k, n)] - u_exact_ijk;
        norm_error += diff*diff;
      }
    }
  }
  norm_error = sqrt(norm_error);
  norm_u_exact = sqrt(norm_u_exact);

  const double relative_error = norm_error/norm_u_exact;
  printf("Relative error after %d iterations: %e\n", num_iter, relative_error);

  free(u);
  free(f);
  return 0;
}
