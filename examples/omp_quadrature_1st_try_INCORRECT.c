#include <stdio.h>
#include <math.h>
#include <omp.h>

const double PI = 3.141592653589793;

double f_test(double x)
{
  return sin(x) + x;
}

double quad_trapezoidal(double (*f)(double), double a, double b, int n)
{
  const double h = (b - a)/n;

  double sum = 0.0;

  #pragma omp parallel
  {
    const int thread_id = omp_get_thread_num();
    const int num_threads = omp_get_num_threads();

    int n_thread = n/num_threads;
    int i_start = thread_id*n_thread;
    if (thread_id < n%num_threads) {
      n_thread++;
      i_start += thread_id;
    }
    else {
      i_start += n%num_threads;
    }
    int i_end = i_start + n_thread - 1;

    for (int i = i_start; i <= i_end; i++) {
      // INCORRECT: DATA RACE!
      sum += 0.5*(f(a + i*h) + f(a + (i + 1)*h));
    }
  }

  return h*sum;
}

int main()
{
  const int n[] = {4, 16, 32, 64, 128, 256, 512};
  const double I_true = 1.0 + PI*PI/8.0;

  for (int j = 0; j < sizeof(n)/sizeof(int); j++) {
    const double I_trapezoidal = quad_trapezoidal(f_test, 0.0, 0.5*PI, n[j]);
    const double error = fabs(I_trapezoidal - I_true);

    printf("n = %d:\n", n[j]);
    printf("Result with trapezoidal rule: %e\n", I_trapezoidal);
    printf("True solution:                %e\n", I_true);
    printf("Relative error:               %e\n", error);
    printf("\n");
  }

  return 0;
}
