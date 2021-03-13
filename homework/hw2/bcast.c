#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>

void my_bcast(void *buf, int count, MPI_Datatype datatype, MPI_Comm comm)
{
  //////////////////////////////////////////////////////////////////////////////
  //                         Add your code here!                              //
  //////////////////////////////////////////////////////////////////////////////
}

void my_bcast_tree(void *buf, int count, MPI_Datatype datatype, MPI_Comm comm)
{
  //////////////////////////////////////////////////////////////////////////////
  //                         Add your code here!                              //
  //////////////////////////////////////////////////////////////////////////////
}

// Do not modify the main function!
int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  srand(time(NULL));

  int seed, result, reference;
  if (rank == 0) seed = rand()%100;
  MPI_Bcast(&seed, 1, MPI_INT, 0, MPI_COMM_WORLD);
  seed += rank;
  MPI_Reduce(&seed, &result, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
  reference = 0;
  for (int r = 0; r < size; r++) {
    reference += seed + r;
  }
  if (rank == 0) {
    if (result == reference) {
      printf("MPI_Bcast() works correctly!\n");
    }
    else {
      printf("MPI_Bcast() failed!\n");
    }
  }

  if (rank == 0) seed = rand()%100;
  my_bcast(&seed, 1, MPI_INT, MPI_COMM_WORLD);
  seed += rank;
  MPI_Reduce(&seed, &result, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
  reference = 0;
  for (int r = 0; r < size; r++) {
    reference += seed + r;
  }
  if (rank == 0) {
    if (result == reference) {
      printf("my_bcast() works correctly!\n");
    }
    else {
      printf("my_bcast() failed!\n");
    }
  }

  if (rank == 0) seed = rand()%100;
  my_bcast_tree(&seed, 1, MPI_INT, MPI_COMM_WORLD);
  seed += rank;
  MPI_Reduce(&seed, &result, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
  reference = 0;
  for (int r = 0; r < size; r++) {
    reference += seed + r;
  }
  if (rank == 0) {
    if (result == reference) {
      printf("my_bcast_tree() works correctly!\n");
    }
    else {
      printf("my_bcast_tree() failed!\n");
    }
  }

  MPI_Finalize();
  return 0;
}
