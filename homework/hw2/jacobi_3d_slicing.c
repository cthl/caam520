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

// Each MPI rank works on a part of the grid that is of size
// ni_loc x nj_loc x nk_loc.
// Since there are ghost layers in positive and negative i-, j-, and
// k-direction, the solution is stored in a *one-dimensional* array of
// dimensions (ni_loc + 2) x (nj_loc + 2) x (nk_loc + 2).
// Hence, we must map each index tuple (i_loc, j_loc, k_loc) to a location in
// the one-dimensional array.
// We do this mapping in a way that allows us to access the ghost layer in
// negative i-direction using i_loc = -1, while the ghost layer in positive
// k-direction is accessed using k_loc = nk_loc, etc.
int ijk_to_index(int i_loc, int j_loc, int k_loc, int ni_loc, int nj_loc)
{
  return (ni_loc + 2)*(nj_loc + 2)*(k_loc + 1)
         + (ni_loc + 2)*(j_loc + 1)
         + (i_loc + 1);
}

int part_to_rank(int pi, int pj, int pk, int npi, int npj, int npk)
{
  //////////////////////////////////////////////////////////////////////////////
  //                         Add your code here!                              //
  //////////////////////////////////////////////////////////////////////////////
}

void rank_to_part(int rank, int npi, int npj, int npk,
                  int *pi, int *pj, int *pk)
{
  //////////////////////////////////////////////////////////////////////////////
  //                         Add your code here!                              //
  //////////////////////////////////////////////////////////////////////////////
}

void jacobi_update(double *u_new,
                   double *u_old,
                   const double *f,
                   int n,
                   int ni_loc, int nj_loc, int nk_loc,
                   int pi, int pj, int pk,
                   int npi, int npj, int npk)
{
  const double h = 1.0/(n + 1);
  const double h2 = h*h;

  //////////////////////////////////////////////////////////////////////////////
  //                    Add your halo exchange code here!                     //
  //              (Feel free to add more functions as needed.)                //
  //////////////////////////////////////////////////////////////////////////////

  // Copy u_new to u_old.
  memcpy(u_old, u_new, (ni_loc + 2)*(nj_loc + 2)*(nk_loc + 2)*sizeof(double));

  // Perform the Jacobi update.
  for (int k_loc = 0; k_loc < nk_loc; k_loc++) {
    for (int j_loc = 0; j_loc < nj_loc; j_loc++) {
      for (int i_loc = 0; i_loc < ni_loc; i_loc++) {
        const int idx = ijk_to_index(i_loc, j_loc, k_loc, ni_loc, nj_loc);

        u_new[idx]
          = (h2*f[idx]
            + u_old[ijk_to_index(i_loc - 1, j_loc, k_loc, ni_loc, nj_loc)]
            + u_old[ijk_to_index(i_loc + 1, j_loc, k_loc, ni_loc, nj_loc)]
            + u_old[ijk_to_index(i_loc, j_loc - 1, k_loc, ni_loc, nj_loc)]
            + u_old[ijk_to_index(i_loc, j_loc + 1, k_loc, ni_loc, nj_loc)]
            + u_old[ijk_to_index(i_loc, j_loc, k_loc - 1, ni_loc, nj_loc)]
            + u_old[ijk_to_index(i_loc, j_loc, k_loc + 1, ni_loc, nj_loc)]
            )/6.0;
      }
    }
  }
}

// Do not modify the main function!
int main(int argc, char **argv)
{
  if (argc != 6) {
    fprintf(stderr,
            "Usage: mpirun -n N ./jacobi_3d_slicing n npi npj npk num_iter\n");
    return -1;
  }
  // Get the problem size.
  const int n = atoi(argv[1]);
  // Get the number of parts (slices) in i-, j-, and k-direction.
  const int npi = atoi(argv[2]);
  const int npj = atoi(argv[3]);
  const int npk = atoi(argv[4]);
  // Get the number of Jacobi iterations.
  const int num_iter = atoi(argv[5]);

  // Ensure that all dimensions work out.
  if (n%npi != 0) {
    fprintf(stderr,
            "Error: Cannot slice %d grid points into %d parts of equal size!\n",
            n, npi);
    return -1;
  }
  if (n%npj != 0) {
    fprintf(stderr,
            "Error: Cannot slice %d grid points into %d parts of equal size!\n",
            n, npj);
    return -1;
  }
  if (n%npk != 0) {
    fprintf(stderr,
            "Error: Cannot slice %d grid points into %d parts of equal size!\n",
            n, npk);
    return -1;
  }

  const double h = 1.0/(n + 1);

  MPI_Init(&argc, &argv);

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // Ensure that we have as many parts as there are MPI ranks.
  if (size != npi*npj*npk) {
    fprintf(stderr,
            "Error: Number of MPI ranks must match the number of parts!\n");
    MPI_Finalize();
    return -1;
  }

  // Compute the extent of each part in i-, j-, and k-direction.
  const int ni_loc = n/npi;
  const int nj_loc = n/npj;
  const int nk_loc = n/npk;

  // Allocate arrays for the new and old solution as well as the right-hand
  // side, including "ghost layers."
  double *u_new = calloc((ni_loc + 2)*(nj_loc + 2)*(nk_loc + 2),
                         sizeof(double));
  double *u_old = calloc((ni_loc + 2)*(nj_loc + 2)*(nk_loc + 2),
                         sizeof(double));
  double *f = malloc((ni_loc + 2)*(nj_loc + 2)*(nk_loc + 2)*sizeof(double));
  if (!u_new || !u_old || !f) {
    free(u_new);
    free(u_old);
    free(f);
    fprintf(stderr, "Error: Could not allocate memory!\n");
    MPI_Finalize();
    return -1;
  }

  // Determine the offsets of this rank's part of the grid.
  int pi, pj, pk;
  rank_to_part(rank, npi, npj, npk, &pi, &pj, &pk);
  const int i_offset = pi*ni_loc;
  const int j_offset = pj*nj_loc;
  const int k_offset = pk*nk_loc;

  // Set initial guess and right-hand side.
  for (int k_loc = 0; k_loc < nk_loc; k_loc++) {
    for (int j_loc = 0; j_loc < nj_loc; j_loc++) {
      for (int i_loc = 0; i_loc < ni_loc; i_loc++) {
        // Compute *global* indices in i-, j-, and k-direction for the current
        // grid point.
        const int i = i_loc + i_offset;
        const int j = j_loc + j_offset;
        const int k = k_loc + k_offset;

        // Compute the coordinates for the current grid point.
        const double x = (i + 1)*h;
        const double y = (j + 1)*h;
        const double z = (k + 1)*h;

        // Compute the index of the current grid point.
        const int idx = ijk_to_index(i_loc, j_loc, k_loc, ni_loc, nj_loc);

        // Set right-hand side.
        f[idx] = f_rhs(x, y, z);
      }
    }
  }

  // Perform Jacobi iteration.
  for (int iter = 0; iter < num_iter; iter++) {
    jacobi_update(u_new, u_old, f,
                  n, ni_loc, nj_loc, nk_loc,
                  pi, pj, pk, npi, npj, npk);
  }

  // Compute relative error.
  double norm_error_loc = 0.0;
  double norm_u_exact_loc = 0.0;
  for (int k_loc = 0; k_loc < nk_loc; k_loc++) {
    for (int j_loc = 0; j_loc < nj_loc; j_loc++) {
      for (int i_loc = 0; i_loc < ni_loc; i_loc++) {
        // Compute *global* indices in i-, j-, and k-direction for the current
        // grid point.
        const int i = i_loc + i_offset;
        const int j = j_loc + j_offset;
        const int k = k_loc + k_offset;

        // Compute the coordinates for the current grid point.
        const double x = (i + 1)*h;
        const double y = (j + 1)*h;
        const double z = (k + 1)*h;

        // Evaluate exact solution.
        const double u_exact_ijk = u_exact(x, y, z);

        // Compute the index of the current grid point.
        const int idx = ijk_to_index(i_loc, j_loc, k_loc, ni_loc, nj_loc);

        // Compute error norm at the current grid point.
        const double diff = u_new[idx] - u_exact_ijk;
        norm_error_loc += diff*diff;
        norm_u_exact_loc += u_exact_ijk*u_exact_ijk;
      }
    }
  }

  double norm_error, norm_u_exact;
  MPI_Allreduce(&norm_error_loc, &norm_error,
                1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  norm_error = sqrt(norm_error);
  MPI_Allreduce(&norm_u_exact_loc, &norm_u_exact,
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
