// metis_jacobi.c - Jacobi update on a METIS-partitioned domain

#include <stdio.h>
#include <stdlib.h>
#include <string.h> // for memcpy()
#include <math.h>
#include <metis.h>
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

// Data structure that contains all grid information.
typedef struct
{
  // The mapping grid point -> MPI rank obtained from METIS
  int *parts;
  // The number of grid points needed from each rank.
  // The entry corresponding to the current rank is simply the number of
  // local grid points, i.e., grid points "owned" by this rank.
  // For all other ranks, this is the number of points they contribute to the
  // ghost layers on this rank.
  int *num_pts_from_rank;
  // An array identical to num_pts_from_rank, except that the entry for the
  // current rank is zero because the rank does not receive its own local
  // values during MPI communication.
  int *recvcounts;
  // Solution vectors are stored in a one-dimensional array with following
  // layout:
  // | ghost values from rank 0    | ... | values on this rank               | ... | ghost values from last rank             |
  // | (size num_pts_from_rank[0]) | ... | (size num_pts_from_rank[my_rank]) | ... | (size num_pts_from_rank[comm_size - 1]) |
  // | (offset offsets[0])         | ... | (offset offsets[my_rank])         | ... | (offset offsets[comm_size - 1])         |
  int *offsets;
  // Total length of the array described above
  int length_with_ghosts;
  // Index conversions
  int *local_to_global;
  int *global_to_local;
  // Number of point values to send to each rank
  int *num_pts_to_rank;
  // Buffer for the local values during MPI communication
  double *sendbuf;
  // Offsets within the send buffer.
  // For example, rank r recives num_pts_to_rank[r] values starting at
  // sendbuf[send_offsets[r]].
  int *send_offsets;
  // Total number of points to send to all ranks combined
  int num_pts_to_send;
  // Local indices of the grid points whose values need to be sent.
  // The value at the grid point with local index send_indices[i] will be
  // copied to sendbuf[i].
  int *send_indices;
  int *local_neighbors;
} grid_t;

// Define an enumeration of the (global) grid points, i.e., assign a unique
// global index to the grid point with coordinates i, j, and k.
int ijk_to_global_index(int i, int j, int k, int n)
{
  return n*n*k + n*j + i;
}

// Define the inverse mapping of ijk_to_global_index().
void global_index_to_ijk(int index, int n, int *i, int *j, int *k)
{
  *i = index%n;
  index /= n;
  *j = index%n;
  index /= n;
  *k = index;
}

// Partition the n x n x n grid and construct all data structures.
void grid_init(grid_t *grid, int n)
{
  int rank, comm_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

  // Prepare call to METIS (cf. metis_example2.c).
  int *adjncy = calloc(6*n*n*n, sizeof(int));
  int *xadj = calloc(n*n*n + 1, sizeof(int));
  int nvtxs = n*n*n;
  int ncon = 1;
  int nparts = comm_size;
  int objval;

  // Construct adjacency graph for the cubical grid.
  int vertex = 0;
  int offset = 0;
  for (int k = 0; k < n; k++) {
    for (int j = 0; j < n; j++) {
      for (int i = 0; i < n; i++) {
        xadj[vertex + 1] = xadj[vertex];
        if (i > 0) {
          adjncy[offset] = ijk_to_global_index(i - 1, j, k, n);
          offset++;
          xadj[vertex + 1]++;
        }
        if (i < n - 1) {
          adjncy[offset] = ijk_to_global_index(i + 1, j, k, n);
          offset++;
          xadj[vertex + 1]++;
        }
        if (j > 0) {
          adjncy[offset] = ijk_to_global_index(i, j - 1, k, n);
          offset++;
          xadj[vertex + 1]++;
        }
        if (j < n - 1) {
          adjncy[offset] = ijk_to_global_index(i, j + 1, k, n);
          offset++;
          xadj[vertex + 1]++;
        }
        if (k > 0) {
          adjncy[offset] = ijk_to_global_index(i, j, k - 1, n);
          offset++;
          xadj[vertex + 1]++;
        }
        if (k < n - 1) {
          adjncy[offset] = ijk_to_global_index(i, j, k + 1, n);
          offset++;
          xadj[vertex + 1]++;
        }
        vertex++;
      }
    }
  }
  xadj[vertex] = offset;

  // After the partitioning, part[i] will tell us
  // which part (subgraph) contains the ith vertex.
  grid->parts = calloc(nvtxs, sizeof(int));

  // Set METIS options.
  int options[METIS_NOPTIONS];
  METIS_SetDefaultOptions(options);
  // Make sure that all MPI ranks use the same seed for the random number
  // generator, i.e., make sure that they end up with the same partitioning.
  options[METIS_OPTION_SEED] = 0;

  // Call METIS.
  const int err = METIS_PartGraphKway(&nvtxs,
                                      &ncon,
                                      xadj,
                                      adjncy,
                                      NULL,
                                      NULL,
                                      NULL,
                                      &nparts,
                                      NULL,
                                      NULL,
                                      options,
                                      &objval,
                                      grid->parts);
  if (err != METIS_OK) {
    fprintf(stderr, "METIS error!\n");
  }

  // Count the number of grid points that we need to send to and receive from
  // each rank.
  grid->num_pts_from_rank = calloc(comm_size, sizeof(int));
  grid->num_pts_to_rank = calloc(comm_size, sizeof(int));

  int *send_to = calloc(comm_size, sizeof(int));
  int global_index = 0;
  for (int k = 0; k < n; k++) {
    for (int j = 0; j < n; j++) {
      for (int i = 0; i < n; i++) {
        int neighbor_rank;

        if (grid->parts[global_index] == rank) {
          // The point with coordinates i, j, and k is a local point,
          // i.e., it has been assigned to the current rank.
          grid->num_pts_from_rank[rank]++;

          // Check which ranks we need to send this point to.
          if (i > 0) {
            neighbor_rank = grid->parts[ijk_to_global_index(i - 1, j, k, n)];
            if (neighbor_rank != rank) send_to[neighbor_rank] = 1;
          }
          if (i < n - 1) {
            neighbor_rank = grid->parts[ijk_to_global_index(i + 1, j, k, n)];
            if (neighbor_rank != rank) send_to[neighbor_rank] = 1;
          }
          if (j > 0) {
            neighbor_rank = grid->parts[ijk_to_global_index(i, j - 1, k, n)];
            if (neighbor_rank != rank) send_to[neighbor_rank] = 1;
          }
          if (j < n - 1) {
            neighbor_rank = grid->parts[ijk_to_global_index(i, j + 1, k, n)];
            if (neighbor_rank != rank) send_to[neighbor_rank] = 1;
          }
          if (k > 0) {
            neighbor_rank = grid->parts[ijk_to_global_index(i, j, k - 1, n)];
            if (neighbor_rank != rank) send_to[neighbor_rank] = 1;
          }
          if (k < n - 1) {
            neighbor_rank = grid->parts[ijk_to_global_index(i, j, k + 1, n)];
            if (neighbor_rank != rank) send_to[neighbor_rank] = 1;
          }

          for (int r = 0; r < comm_size; r++) {
            if (send_to[r]) {
              grid->num_pts_to_rank[r]++;
              // Reset flag.
              send_to[r] = 0;
            }
          }
        }
        else {
          // The current grid point is *not* owned by this rank.
          // Check if we need to receive it as part of a ghost layer.
          int recv = 0;

          if (i > 0) {
            // The current grid point has a neighbor in negative i-direction.
            // If that neighbor (i - 1, j, k) is owned by the current rank,
            // then the point (i, j, k) must be part of the ghost layer on
            // the current rank. 
            neighbor_rank = grid->parts[ijk_to_global_index(i - 1, j, k, n)];
            if (neighbor_rank == rank) {
              // The point is part of the ghost layer on this rank, i.e., we
              // receive one more point from the rank specified by
              // parts[global_index].
              recv = 1;
            }
          }
          // etc.
          if (i < n - 1) {
            neighbor_rank = grid->parts[ijk_to_global_index(i + 1, j, k, n)];
            if (neighbor_rank == rank) recv = 1;
          }
          if (j > 0) {
            neighbor_rank = grid->parts[ijk_to_global_index(i, j - 1, k, n)];
            if (neighbor_rank == rank) recv = 1;
          }
          if (j < n - 1) {
            neighbor_rank = grid->parts[ijk_to_global_index(i, j + 1, k, n)];
            if (neighbor_rank == rank) recv = 1;
          }
          if (k > 0) {
            neighbor_rank = grid->parts[ijk_to_global_index(i, j, k - 1, n)];
            if (neighbor_rank == rank) recv = 1;
          }
          if (k < n - 1) {
            neighbor_rank = grid->parts[ijk_to_global_index(i, j, k + 1, n)];
            if (neighbor_rank == rank) recv = 1;
          }

          if (recv) {
            grid->num_pts_from_rank[grid->parts[global_index]]++;
          }
        }
        global_index++;
      }
    }
  }

  // Set up the recvcounts array.
  grid->recvcounts = calloc(comm_size, sizeof(int));
  memcpy(grid->recvcounts, grid->num_pts_from_rank, comm_size*sizeof(int));
  grid->recvcounts[rank] = 0;

  // Compute offsets and lengths (cf. grid_t).
  grid->offsets = calloc(comm_size, sizeof(int));
  grid->send_offsets = calloc(comm_size, sizeof(int));
  for (int r = 1; r < comm_size; r++) {
    grid->offsets[r] = grid->offsets[r - 1] + grid->num_pts_from_rank[r - 1];
    grid->send_offsets[r] = grid->send_offsets[r - 1]
                          + grid->num_pts_to_rank[r - 1];
  }
  grid->length_with_ghosts = grid->offsets[comm_size - 1]
                           + grid->num_pts_from_rank[comm_size - 1];
  grid->num_pts_to_send = grid->send_offsets[comm_size - 1]
                        + grid->num_pts_to_rank[comm_size - 1];

  // Allocate the send buffer and the array of local indices to send.
  grid->send_indices = calloc(grid->num_pts_to_send, sizeof(int));
  grid->sendbuf = calloc(grid->num_pts_to_send, sizeof(double));

  // Set up the conversion between local and global grid point indices.
  grid->local_to_global = calloc(grid->length_with_ghosts, sizeof(int));
  grid->global_to_local = calloc(n*n*n, sizeof(int));
  int *pos = calloc(comm_size, sizeof(int));
  int *send_pos = calloc(comm_size, sizeof(int));

  global_index = 0;
  for (int k = 0; k < n; k++) {
    for (int j = 0; j < n; j++) {
      for (int i = 0; i < n; i++) {
        int neighbor_rank;
        // Determine the MPI rank that owns the current grid point.
        const int point_rank = grid->parts[global_index];

        if (point_rank == rank) {
          // The grid point is owned by the current rank.
          const int local_index = grid->offsets[rank] + pos[rank];
          grid->local_to_global[local_index] = global_index;
          grid->global_to_local[global_index] = local_index;
          pos[rank]++;

          // Keep track of which ranks we need to send this point to during the
          // halo exchange.
          if (i > 0) {
            neighbor_rank = grid->parts[ijk_to_global_index(i - 1, j, k, n)];
            if (neighbor_rank != rank) send_to[neighbor_rank] = 1;
          }
          if (i < n - 1) {
            neighbor_rank = grid->parts[ijk_to_global_index(i + 1, j, k, n)];
            if (neighbor_rank != rank) send_to[neighbor_rank] = 1;
          }
          if (j > 0) {
            neighbor_rank = grid->parts[ijk_to_global_index(i, j - 1, k, n)];
            if (neighbor_rank != rank) send_to[neighbor_rank] = 1;
          }
          if (j < n - 1) {
            neighbor_rank = grid->parts[ijk_to_global_index(i, j + 1, k, n)];
            if (neighbor_rank != rank) send_to[neighbor_rank] = 1;
          }
          if (k > 0) {
            neighbor_rank = grid->parts[ijk_to_global_index(i, j, k - 1, n)];
            if (neighbor_rank != rank) send_to[neighbor_rank] = 1;
          }
          if (k < n - 1) {
            neighbor_rank = grid->parts[ijk_to_global_index(i, j, k + 1, n)];
            if (neighbor_rank != rank) send_to[neighbor_rank] = 1;
          }

          for (int r = 0; r < comm_size; r++) {
            if (send_to[r]) {
              const int s = grid->send_offsets[r] + send_pos[r];
              grid->send_indices[s] = local_index;
              send_pos[r]++;
              // Reset flag.
              send_to[r] = 0;
            }
          }
        }
        else {
          // The grid point is owned by another rank.
          // We need to check if it is part of a ghost layer on the current
          // rank, i.e., if the grid point is adjacent to this rank.
          int recv = 0;
          // This will be the point's local index *if* it is added to one of
          // the ghost layers.
          const int local_index = grid->offsets[point_rank] + pos[point_rank];

          if (i > 0) {
            neighbor_rank = grid->parts[ijk_to_global_index(i - 1, j, k, n)];
            if (neighbor_rank == rank) recv = 1;
          }
          if (i < n - 1) {
            neighbor_rank = grid->parts[ijk_to_global_index(i + 1, j, k, n)];
            if (neighbor_rank == rank) recv = 1;
          }
          if (j > 0) {
            neighbor_rank = grid->parts[ijk_to_global_index(i, j - 1, k, n)];
            if (neighbor_rank == rank) recv = 1;
          }
          if (j < n - 1) {
            neighbor_rank = grid->parts[ijk_to_global_index(i, j + 1, k, n)];
            if (neighbor_rank == rank) recv = 1;
          }
          if (k > 0) {
            neighbor_rank = grid->parts[ijk_to_global_index(i, j, k - 1, n)];
            if (neighbor_rank == rank) recv = 1;
          }
          if (k < n - 1) {
            neighbor_rank = grid->parts[ijk_to_global_index(i, j, k + 1, n)];
            if (neighbor_rank == rank) recv = 1;
          }

          if (recv) {
            grid->local_to_global[local_index] = global_index;
            grid->global_to_local[global_index] = local_index;
            pos[point_rank]++;
          }
          else {
            // The current grid point is not part of a ghost layer.
            // Hence, it does not have a valid local index.
            grid->global_to_local[global_index] = -999;
          }
        }
        global_index++;
      }
    }
  }

  grid->local_neighbors = calloc(6*grid->length_with_ghosts, sizeof(int));

  for (int s = 0; s < 6*grid->length_with_ghosts; s++) {
    grid->local_neighbors[s] = -999;
  }

  for (int local_index = grid->offsets[rank];
       local_index < grid->offsets[rank] + grid->num_pts_from_rank[rank];
       local_index++) {
    // Compute the global index and the i, j, and k coordinates.
    int i, j, k;
    const int global_index = grid->local_to_global[local_index];
    global_index_to_ijk(global_index, n, &i, &j, &k);

    int global_index_neighbor, local_index_neighbor;
    if (i > 0) {
      global_index_neighbor = ijk_to_global_index(i - 1, j, k, n);
      local_index_neighbor = grid->global_to_local[global_index_neighbor];
      grid->local_neighbors[6*local_index + 0] = local_index_neighbor;
    }
    if (i < n - 1) {
      global_index_neighbor = ijk_to_global_index(i + 1, j, k, n);
      local_index_neighbor = grid->global_to_local[global_index_neighbor];
      grid->local_neighbors[6*local_index + 1] = local_index_neighbor;
    }
    if (j > 0) {
      global_index_neighbor = ijk_to_global_index(i, j - 1, k, n);
      local_index_neighbor = grid->global_to_local[global_index_neighbor];
      grid->local_neighbors[6*local_index + 2] = local_index_neighbor;
    }
    if (j < n - 1) {
      global_index_neighbor = ijk_to_global_index(i, j + 1, k, n);
      local_index_neighbor = grid->global_to_local[global_index_neighbor];
      grid->local_neighbors[6*local_index + 3] = local_index_neighbor;
    }
    if (k > 0) {
      global_index_neighbor = ijk_to_global_index(i, j, k - 1, n);
      local_index_neighbor = grid->global_to_local[global_index_neighbor];
      grid->local_neighbors[6*local_index + 4] = local_index_neighbor;
    }
    if (k < n - 1) {
      global_index_neighbor = ijk_to_global_index(i, j, k + 1, n);
      local_index_neighbor = grid->global_to_local[global_index_neighbor];
      grid->local_neighbors[6*local_index + 5] = local_index_neighbor;
    }
  }

  // Clean up.
  free(send_to);
  free(pos);
  free(send_pos);
  free(xadj);
  free(adjncy);
}

void grid_free(grid_t *grid)
{
  free(grid->local_neighbors);
  free(grid->send_indices);
  free(grid->send_offsets);
  free(grid->sendbuf);
  free(grid->num_pts_to_rank);
  free(grid->recvcounts);
  free(grid->global_to_local);
  free(grid->local_to_global);
  free(grid->num_pts_from_rank);
  free(grid->offsets);
  free(grid->parts);
}

void exchange_halo(double *u, const grid_t *grid)
{
  for (int s = 0; s < grid->num_pts_to_send; s++) {
    grid->sendbuf[s] = u[grid->send_indices[s]];
  }

  MPI_Alltoallv(grid->sendbuf,
                grid->num_pts_to_rank,
                grid->send_offsets,
                MPI_DOUBLE,
                u,
                grid->recvcounts,
                grid->offsets,
                MPI_DOUBLE,
                MPI_COMM_WORLD);
}

void jacobi_update(double *u_new,
                   double *u_old,
                   const double *f,
                   int n,
                   const grid_t *grid)
{
  const double h = 1.0/(n + 1);
  const double h2 = h*h;

  int rank, comm_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

  // Perform a halo exchange, i.e., get ghost layer values from neighboring
  // ranks.
  exchange_halo(u_new, grid);

  // Copy u_new to u_old.
  memcpy(u_old, u_new, grid->length_with_ghosts*sizeof(double));

  // Loop over all *local* grid points to perform the Jacobi update.
  for (int local_index = grid->offsets[rank];
       local_index < grid->offsets[rank] + grid->num_pts_from_rank[rank];
       local_index++) {
    // Compute the global index and the i, j, and k coordinates.
    int i, j, k;

    double u_ijk = h2*f[local_index];

    for (int d = 0; d < 6; d++) {
      const int local_neighbor = grid->local_neighbors[6*local_index + d];
      if (local_neighbor >= 0) {
        u_ijk += u_old[local_neighbor];
      }
    }

    u_ijk /= 6.0;
    u_new[local_index] = u_ijk;
  }
}

int main(int argc, char **argv)
{
  // Get problem size and number of iterations.
  if (argc < 3) {
    fprintf(stderr, "Usage: mpirun -n N ./metis_jacobi n num_iter [export]\n");
    return -1;
  }
  const int n = atoi(argv[1]);
  const int num_iter = atoi(argv[2]);
  int export = 0;
  if (argc > 3) {
    export = atoi(argv[3]);
  }

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

  grid_t grid;
  grid_init(&grid, n);

  // Allocate arrays for the new and old solution as well as the right-hand
  // side.
  // Include "ghost layers."
  double *u_new = calloc(grid.length_with_ghosts, sizeof(double));
  double *u_old = calloc(grid.length_with_ghosts, sizeof(double));
  double *f = calloc(grid.length_with_ghosts, sizeof(double));

  // Set initial guess and right-hand side.
  for (int local_index = grid.offsets[rank];
       local_index < grid.offsets[rank] + grid.num_pts_from_rank[rank];
       local_index++) {
    // Compute the global index and the i, j, and k coordinates.
    int i, j, k;
    const int global_index = grid.local_to_global[local_index];
    global_index_to_ijk(global_index, n, &i, &j, &k);

    // Compute the coordinates for the current grid point.
    const double x = (i + 1)*h;
    const double y = (j + 1)*h;
    const double z = (k + 1)*h;

    // Set initial guess.
    u_old[local_index] = 0.0;

    // Set right-hand side.
    f[local_index] = f_rhs(x, y, z);
  }

  // Perform Jacobi iteration.
  for (int iter = 0; iter < num_iter; iter++) {
    jacobi_update(u_new, u_old, f, n, &grid);
  }

  // Compute relative error.
  double norm_error_local = 0.0;
  double norm_u_exact_local = 0.0;
  for (int local_index = grid.offsets[rank];
       local_index < grid.offsets[rank] + grid.num_pts_from_rank[rank];
       local_index++) {
    // Compute the global index and the i, j, and k coordinates.
    int i, j, k;
    const int global_index = grid.local_to_global[local_index];
    global_index_to_ijk(global_index, n, &i, &j, &k);

    // Compute the coordinates for the current grid point.
    const double x = (i + 1)*h;
    const double y = (j + 1)*h;
    const double z = (k + 1)*h;

    // Evaluate exact solution.
    const double u_exact_ijk = u_exact(x, y, z);

    // Compute error norm at the current grid point.
    const double diff = u_new[local_index] - u_exact_ijk;
    norm_error_local += diff*diff;
    norm_u_exact_local += u_exact_ijk*u_exact_ijk;
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

  if (export) {
    FILE *f;
    f = fopen("grid.bin", "w");
    fwrite(grid.parts, sizeof(int), n*n*n, f);
    fclose(f);
  }

  grid_free(&grid);
  free(u_new);
  free(u_old);
  free(f);
  MPI_Finalize();
  return 0;
}
