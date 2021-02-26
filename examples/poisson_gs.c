// poisson_gs.c - Solve Poisson's equation using finite differences
//                and the Gauss-Seidel method.

#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#define M_PI 3.141592653589793
#define IJK2INDEX(I, J, K, N) (N*N*K + N*J + I)

double u_exact(double x, double y, double z)
{
  return sin(M_PI*x)*sin(M_PI*y)*sin(M_PI*z);
}

double f_rhs(double x, double y, double z)
{
  return 3.0*M_PI*M_PI*u_exact(x, y, z);
}

void gauss_seidel(double *u, const double *f, int n, int num_iter)
{
  const double h = 1.0/(n + 1);
  const double h2 = h*h;

  for (int iter = 0; iter < num_iter; iter++) {
    for (int k = 0; k < n; k++) {
      for (int j = 0; j < n; j++) {
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
        }
      }
    }
  }
}

int main(int argc, char **argv)
{
  if (argc != 3) {
    fprintf(stderr, "Usage: ./poisson_gs n num_iter\n");
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
