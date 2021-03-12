// metis_example1.c - Simple graph partitioning with METIS

// This example partitions the a cube of n x n x n points.

#include <stdio.h>
#include <stdlib.h>
#include <metis.h>

int ijk_to_index(int i, int j, int k, int n)
{
  return n*n*k + n*j + i;
}

int main()
{
  // Number of grid points in each direction.
  const int n = 64;
  // Adjacent vertices for each vertex in the graph
  int *adjncy = calloc(6*n*n*n, sizeof(int));
  // Define each vertex' section in the above array.
  // Compare this to the row_ptr array for CSR matrices.
  int *xadj = calloc(n*n*n + 1, sizeof(int));
  // Number of vertices in the graph
  int nvtxs = n*n*n;
  // Number of constraints
  int ncon = 1;
  // Number of parts after the partitioning
  int nparts = 7;
  // Final objective value
  int objval;

  int vertex = 0;
  int offset = 0;
  for (int k = 0; k < n; k++) {
    for (int j = 0; j < n; j++) {
      for (int i = 0; i < n; i++) {
        xadj[vertex + 1] = xadj[vertex];
        if (i > 0) {
          adjncy[offset] = ijk_to_index(i - 1, j, k, n);
          offset++;
          xadj[vertex + 1]++;
        }
        if (i < n - 1) {
          adjncy[offset] = ijk_to_index(i + 1, j, k, n);
          offset++;
          xadj[vertex + 1]++;
        }
        if (j > 0) {
          adjncy[offset] = ijk_to_index(i, j - 1, k, n);
          offset++;
          xadj[vertex + 1]++;
        }
        if (j < n - 1) {
          adjncy[offset] = ijk_to_index(i, j + 1, k, n);
          offset++;
          xadj[vertex + 1]++;
        }
        if (k > 0) {
          adjncy[offset] = ijk_to_index(i, j, k - 1, n);
          offset++;
          xadj[vertex + 1]++;
        }
        if (k < n - 1) {
          adjncy[offset] = ijk_to_index(i, j, k + 1, n);
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

  // Export partitioning to a binary file.
  FILE *f = fopen("part.bin", "w");
  fwrite(part, sizeof(int), nvtxs, f);
  fclose(f);

  free(part);
  free(xadj);
  free(adjncy);
  return 0;
}
