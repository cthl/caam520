// mpi_ring.c - Send data around in an MPI communicator using a ring structure.

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char **argv)
{
  // Get number of rounds.
  if (argc != 2) {
    fprintf(stderr, "Usage: mpirun -n N ./mpi_ring num_rounds\n");
  }
  const int num_rounds = atoi(argv[1]);

  MPI_Init(&argc, &argv);

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size < 2) {
    fprintf(stderr, "Error: Program must be run with at least two ranks!\n");
    MPI_Finalize();
    return -1;
  }

  int send, recv;
  for (int r = 0; r < num_rounds; r++) {
    // Send data to next rank and receive data from previous rank.
    // The first rank receives data from the last rank.
    const int to = (rank + 1)%size;
    const int from = rank == 0 ? size - 1 : rank - 1;

    MPI_Sendrecv(&send, 1, MPI_INT, to, 999,
                 &recv, 1, MPI_INT, from, 999,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    printf("#%d: sent data to rank %d, received data from rank %d\n",
           rank, to, from);
  }

  MPI_Finalize();
  return 0;
}
