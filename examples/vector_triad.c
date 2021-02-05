#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "time_diff.h"

int main()
{
  // Maximum vector size.
  const int max_size = 1e7;
  // Number of repetitions.
  const int reps = 100;

  double *a = malloc(max_size*sizeof(double));
  double *b = malloc(max_size*sizeof(double));
  double *c = malloc(max_size*sizeof(double));
  double *d = malloc(max_size*sizeof(double));

  const double factor = pow(10, 0.25);
  for (int size = 10; size <= max_size; size *= factor) {
    // Initialize vectors.
    for (int i = 0; i < size; i++) {
      a[i] = 0.0;
      b[i] = 1.0;
      c[i] = 2.0;
      d[i] = 3.0;
    }

    // Start timer.
    struct timespec start;
    clock_gettime(CLOCK_MONOTONIC, &start);

    // Perform repeated computation.
    for (int r = 0; r < reps; r++) {
      for (int i = 0; i < size; i++) {
        a[i] = b[i] + c[i]*d[i];
      }
    }

    // Stop timer.
    struct timespec end;
    clock_gettime(CLOCK_MONOTONIC, &end);

    // Compute and print FLOPS.
    const double mflops = 2.0*reps*size/time_diff(&end, &start)/1000.0;
    printf("%d %f\n", size, mflops);
  }

  free(a);
  free(b);
  free(c);
  free(d);
  return 0;
}

