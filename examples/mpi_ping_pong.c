// mpi_ping_pong.c - Send data back and forth between two MPI ranks.

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char **argv)
{
  // Get number of rounds.
  if (argc != 2) {
    fprintf(stderr, "Usage: mpirun -n 2 ./mpi_ping_pong num_rounds\n");
  }
  const int num_rounds = atoi(argv[1]);

  MPI_Init(&argc, &argv);

  int data, rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size != 2) {
    fprintf(stderr, "Error: Program must be run with exactly two ranks!\n");
    MPI_Finalize();
    return -1;
  }

  for (int r = 0; r < num_rounds; r++) {
    if (r%2 == 0) {
      // Even round: Send from rank zero to rank one.
      if (rank == 0) {
        MPI_Send(&data, 1, MPI_INT, 1, 999, MPI_COMM_WORLD);
        printf("#0: sent data to rank #1\n");
      }
      else {
        MPI_Recv(&data, 1, MPI_INT, 0, 999, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("#1: received data from rank #0\n");
      }
    }
    else {
      // Odd round: Send from rank one to rank zero.
      if (rank == 0) {
        MPI_Recv(&data, 1, MPI_INT, 1, 999, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("#0: received data from rank #1\n");
      }
      else {
        MPI_Send(&data, 1, MPI_INT, 0, 999, MPI_COMM_WORLD);
        printf("#1: sent data to rank #0\n");
      }
    }
  }

  MPI_Finalize();
  return 0;
}
