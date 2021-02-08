#include <stdio.h>
#include <math.h>

const double PI = 3.141592653589793;

double f_test(double x)
{
  return sin(x) + x;
}

double quad_trapezoidal(double (*f)(double), double a, double b, int n)
{
  const double h = (b - a)/n;

  double sum = 0.0;
  for (int i = 0; i < n; i++) {
    sum += 0.5*(f(a + i*h) + f(a + (i + 1)*h));
  }

  return h*sum;
}

int main()
{
  const int n[] = {4, 16, 32, 64, 128, 256};
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
