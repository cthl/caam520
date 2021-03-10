// mpi_poisson_jacobi.c - Solve Poisson's equation using finite differences
//                        and the Jacobi method.

#include <stdio.h>
#include <stdlib.h>
#include <string.h> // for memcpy()
#include <math.h>
#include <mpi.h>

#define M_PI 3.141592653589793

double u_exact(double x, double y, double z)
{
  return sin(M_PI*x)*sin(M_PI*y)*sin(M_PI*z);
}

double f_rhs(double x, double y, double z)
{
  return 3.0*M_PI*M_PI*u_exact(x, y, z);
}

void exchange_halo(double *u, int n, int n_local)
{
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // Get pointers to the upper and lower ghost layers.
  double *lower = u;
  double *upper = &u[n*n*(n_local + 1)];
  // Get pointers to the top and bottom layer of this rank's slice of the grid,
  // i.e., the first and last layer that is *not* a ghost layer.
  double *bottom = &u[n*n];
  double *top = &u[n*n*n_local];

  // Send top layer to next rank, receive lower ghost layer from previous rank.
  // Rank zero only sends data, while the last rank only receives data.
  if (rank == 0) {
    MPI_Send(top, n*n, MPI_DOUBLE, 1, 999, MPI_COMM_WORLD);
  }
  else if (rank == size - 1) {
    MPI_Recv(lower, n*n, MPI_DOUBLE, size - 2, 999,
             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }
  else {
    MPI_Sendrecv(top, n*n, MPI_DOUBLE, rank + 1, 999,
                 lower, n*n, MPI_DOUBLE, rank - 1, 999,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }

  // Send bottom layer to previous rank, receive upper ghost layer from next
  // rank.
  // Rank zero only receives data, while the last rank only sends data.
  if (rank == 0) {
    MPI_Recv(upper, n*n, MPI_DOUBLE, 1, 999, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }
  else if (rank == size - 1) {
    MPI_Send(bottom, n*n, MPI_DOUBLE, size - 2, 999, MPI_COMM_WORLD);
  }
  else {
    MPI_Sendrecv(bottom, n*n, MPI_DOUBLE, rank - 1, 999,
                 upper, n*n, MPI_DOUBLE, rank + 1, 999,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }
}

void jacobi_update(double *u_new,
                   double *u_old,
                   const double *f,
                   int n,
                   int n_local)
{
  const double h = 1.0/(n + 1);
  const double h2 = h*h;

  // Perform a halo exchange, i.e., get ghost layer values from neighboring
  // ranks.
  exchange_halo(u_new, n, n_local);

  // Copy u_new to u_old.
  memcpy(u_old, u_new, n*n*(n_local + 2)*sizeof(double));

  // Perform the Jacobi update.
  for (int k_local = 0; k_local < n_local; k_local++) {
    for (int j = 0; j < n; j++) {
      for (int i = 0; i < n; i++) {
        // Compute the index of the current grid point, offset by the ghost
        // layer size (n x n).
        const int idx = n*n*k_local + n*j + i + n*n;

        double u_ijk = h2*f[idx];
        if (i > 0)     u_ijk += u_old[idx - 1];
        if (i < n - 1) u_ijk += u_old[idx + 1];
        if (j > 0)     u_ijk += u_old[idx - n];
        if (j < n - 1) u_ijk += u_old[idx + n];
        // Thanks to the ghost layers, there is always a previous element and 
        // a next element in k-direction.
        u_ijk += u_old[idx - n*n];
        u_ijk += u_old[idx + n*n];

        u_ijk /= 6.0;
        u_new[idx] = u_ijk;
      }
    }
  }
}

int main(int argc, char **argv)
{
  // Get problem size and number of iterations.
  if (argc != 3) {
    fprintf(stderr, "Usage: mpirun -n N ./mpi_poisson_jacobi n num_iter\n");
    return -1;
  }
  const int n = atoi(argv[1]);
  const int num_iter = atoi(argv[2]);

  const double h = 1.0/(n + 1);

  MPI_Init(&argc, &argv);

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (size < 2) {
    fprintf(stderr, "Error: Use at least two MPI ranks!\n");
    MPI_Finalize();
    return -1;
  }

  // We distribute a domain of size n x n x n between all MPI ranks by slicing
  // it along one dimension (vertically, k-direction).
  int n_local = n/size;
  int k_offset = rank*n_local;
  if (rank < n%size) {
    n_local++;
    k_offset += rank;
  }
  else {
    k_offset += n%size;
  }


  // Allocate arrays for the new and old solution as well as the right-hand
  // side.
  // Include "ghost layers."
  double *u_new = malloc(n*n*(n_local + 2)*sizeof(double));
  double *u_old = malloc(n*n*(n_local + 2)*sizeof(double));
  double *f = malloc(n*n*(n_local + 2)*sizeof(double));
  if (!u_new || !u_old || !f) {
    free(u_new);
    free(u_old);
    free(f);
    fprintf(stderr, "Error: Could not allocate memory!\n");
    MPI_Finalize();
    return -1;
  }

  // Set initial guess and right-hand side.
  for (int k_local = 0; k_local < n_local; k_local++) {
    for (int j = 0; j < n; j++) {
      for (int i = 0; i < n; i++) {
        // Compute *global* index in k-direction for the current grid point.
        const int k = k_local + k_offset;

        // Compute the coordinates for the current grid point.
        const double x = (i + 1)*h;
        const double y = (j + 1)*h;
        const double z = (k + 1)*h;

        // Compute the index of the current grid point, offset by the ghost
        // layer size (n x n).
        const int idx = n*n*k_local + n*j + i + n*n;

        // Set initial guess.
        u_old[idx] = 0.0;

        // Set right-hand side.
        f[idx] = f_rhs(x, y, z);
      }
    }
  }

  // Set values in ghost layers to zero.
  // This includes the boundary conditions via the lower ghost layer on rank
  // zero and the upper ghost layer on the last rank.
  for (int j = 0; j < n; j++) {
    for (int i = 0; i < n; i++) {
      // Compute indices in lower and upper ghost layer.
      const int idx_lower = n*j + i;
      const int idx_upper = n*n*(n_local + 1) + n*j + i;

      // Set values.
      u_old[idx_lower] = 0.0;
      u_old[idx_upper] = 0.0;
    }
  }

  // Perform Jacobi iteration.
  for (int iter = 0; iter < num_iter; iter++) {
    jacobi_update(u_new, u_old, f, n, n_local);
  }

  // Compute relative error.
  double norm_error_local = 0.0;
  double norm_u_exact_local = 0.0;
  for (int k_local = 0; k_local < n_local; k_local++) {
    for (int j = 0; j < n; j++) {
      for (int i = 0; i < n; i++) {
        // Compute *global* index in k-direction for the current grid point.
        const int k = k_local + k_offset;

        // Compute the coordinates for the current grid point.
        const double x = (i + 1)*h;
        const double y = (j + 1)*h;
        const double z = (k + 1)*h;

        // Evaluate exact solution.
        const double u_exact_ijk = u_exact(x, y, z);

        // Compute the index of the current grid point, offset by the ghost
        // layer size (n x n).
        const int idx = n*n*k_local + n*j + i + n*n;

        // Compute error norm at the current grid point.
        const double diff = u_new[idx] - u_exact_ijk;
        norm_error_local += diff*diff;
        norm_u_exact_local += u_exact_ijk*u_exact_ijk;
      }
    }
  }

  double norm_error, norm_u_exact;
  MPI_Allreduce(&norm_error_local, &norm_error,
                1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  norm_error = sqrt(norm_error);
  MPI_Allreduce(&norm_u_exact_local, &norm_u_exact,
                1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  norm_u_exact = sqrt(norm_u_exact);

  const double relative_error = norm_error/norm_u_exact;
  if (rank == 0) {
    printf("Relative error after %d iterations: %e\n",
           num_iter, relative_error);
  }

  MPI_Finalize();
  free(u_new);
  free(u_old);
  free(f);
  return 0;
}
