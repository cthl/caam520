// mpi_poisson_gs.c - Solve Poisson's equation using finite differences
//                    and a wave front Gauss-Seidel method.

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

  // Get pointer to the upper ghost layer.
  double *upper = &u[n*n*(n_local + 1)];
  // Get pointer to the bottom layer of this rank's slice of the grid,
  // i.e., the first layer that is *not* a ghost layer.
  double *bottom = &u[n*n];

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

void gauss_seidel_update(double *u,
                         const double *f,
                         int n,
                         int n_local,
                         int block_size)
{
  const double h = 1.0/(n + 1);
  const double h2 = h*h;

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // Perform a halo exchange, i.e., get ghost layer values from neighboring
  // ranks.
  exchange_halo(u, n, n_local);

  // Get pointers to the lower ghost layer and the top layer of this rank's
  // slice of the grid.
  double *lower = u;
  double *top = &u[n*n*n_local];

  // Perform the Gauss-Seidel update.
  for (int j = 0; j < n; j++) {
    if (j%block_size == 0 && rank > 0) {
      // We are entering a new block in j-direction.
      // Get the ghost layer values from the previous rank.
      MPI_Recv(&lower[n*j], n*block_size, MPI_DOUBLE,
               rank - 1, 999, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    for (int k_local = 0; k_local < n_local; k_local++) {
      for (int i = 0; i < n; i++) {
        // Compute the index of the current grid point, offset by the ghost
        // layer size (n x n).
        const int idx = n*n*k_local + n*j + i + n*n;

        double u_ijk = h2*f[idx];
        if (i > 0)     u_ijk += u[idx - 1];
        if (i < n - 1) u_ijk += u[idx + 1];
        if (j > 0)     u_ijk += u[idx - n];
        if (j < n - 1) u_ijk += u[idx + n];
        // Thanks to the ghost layers, there is always a previous element and 
        // a next element in k-direction.
        u_ijk += u[idx - n*n];
        u_ijk += u[idx + n*n];

        u_ijk /= 6.0;
        u[idx] = u_ijk;
      }
    }

    if (j%block_size == block_size - 1 && rank < size - 1) {
      // We finished a block in j-direction.
      // Send updated boundary values to next rank.
      MPI_Send(&top[n*(j - block_size + 1)], n*block_size, MPI_DOUBLE,
               rank + 1, 999, MPI_COMM_WORLD);
    }
  }
}

int main(int argc, char **argv)
{
  // Get problem size, block size, and number of iterations.
  if (argc != 4) {
    fprintf(stderr,
            "Usage: mpirun -n N ./mpi_poisson_gs n block_size num_iter\n");
    return -1;
  }
  const int n = atoi(argv[1]);
  const int block_size = atoi(argv[2]);
  const int num_iter = atoi(argv[3]);

  const double h = 1.0/(n + 1);

  if (n%block_size != 0) {
    fprintf(stderr, "Error: Block size must divide the problem size!\n");
    return -1;
  }

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


  // Allocate arrays for the solution as well as the right-hand side.
  // Include "ghost layers."
  double *u = malloc(n*n*(n_local + 2)*sizeof(double));
  double *f = malloc(n*n*(n_local + 2)*sizeof(double));
  if (!u || !f) {
    free(u);
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
        u[idx] = 0.0;

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
      u[idx_lower] = 0.0;
      u[idx_upper] = 0.0;
    }
  }

  // Perform Gauss-Seidel iteration.
  for (int iter = 0; iter < num_iter; iter++) {
    gauss_seidel_update(u, f, n, n_local, block_size);
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
        const double diff = u[idx] - u_exact_ijk;
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
  free(u);
  free(f);
  return 0;
}
