#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <omp.h>

const double PI = 3.141592653589793;

int main(int argc, char **argv)
{
  const int num_points = 1000000;

  int num_points_in_circle = 0;
  #pragma omp parallel
  {
    unsigned int seed = omp_get_thread_num();

    #pragma omp for reduction(+:num_points_in_circle)
    for (int i = 0; i < num_points; i++) {
      // Generate a random point in [0, 1]^2.
      const double x = ((double) rand_r(&seed))/RAND_MAX;
      const double y = ((double) rand_r(&seed))/RAND_MAX;

      // Count points in the quarter circle.
      if (sqrt(x*x + y*y) < 1.0) {
        num_points_in_circle++;
      }
    }
  }

  // Compute approximation to pi.
  const double pi_approx = 4.0*num_points_in_circle/num_points;

  // Print results.
  printf("Pi:            %f\n", PI);
  printf("Approximation: %f\n", pi_approx);
  printf("Error:         %e\n", fabs(PI - pi_approx));

  return 0;
}
