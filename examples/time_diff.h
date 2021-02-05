#ifndef TIME_DIFF_H
#define TIME_DIFF_H

#include <time.h>

double time_diff(const struct timespec *end, const struct timespec *start)
{
  const double start_double = start->tv_sec + 1.0e-9*start->tv_nsec;
  const double end_double = end->tv_sec + 1.0e-9*end->tv_nsec;

  return end_double - start_double;
}

#endif // TIME_DIFF_H
