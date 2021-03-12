// metis_example1.c - Simple graph partitioning with METIS

// This example partitions the following graph:
//
//  1       3
//   \     / \
//    0 - 2 - 4
//

#include <stdio.h>
#include <stdlib.h>
#include <metis.h>

int main()
{
  // Adjacent vertices for each vertex in the graph
  int adjncy[] = {1, 2, 0, 0, 3, 4, 2, 4, 2, 3};
  // Define each vertex' section in the above array.
  // Compare this to the row_ptr array for CSR matrices.
  int xadj[] = {0, 2, 3, 6, 8, 10};
  // Number of vertices in the graph
  int nvtxs = 5;
  // Number of constraints
  int ncon = 1;
  // Number of parts after the partitioning
  int nparts = 2;
  // Final objective value
  int objval;

  // After the partitioning, part[i] will tell us
  // which part (subgraph) contains the ith vertex.
  int *part = malloc(nvtxs*sizeof(int));

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
                                      NULL,
                                      &objval,
                                      part);
  if (err != METIS_OK) {
    fprintf(stderr, "METIS error!\n");
  }

  // Print the partitioning.
  for (int i = 0; i < nvtxs; i++) {
    printf("Vertex %d -> part %d\n", i, part[i]);
  }

  free(part);
  return 0;
}
