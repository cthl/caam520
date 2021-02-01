// This example is identical to sort.c, except that it sorts large amounts of
// random data, so it can serve as a target for profiling.
// See sort.c for explanations of the algorithms etc.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

typedef char str32[32];

void sort_integer(int *array, int n)
{
  int tmp;

  for (int i = 0; i < n; i++) {
    for (int j = i + 1; j < n; j++) {
      if (array[j] < array[i]) {
        tmp = array[i];
        array[i] = array[j];
        array[j] = tmp;
      }
    }
  }
}

void sort(void *array,
          int n,
          int size,
          int (*compare)(const void*, const void*))
{
  char *array_bytes = (char*) array;

  void *tmp = malloc(size);

  for (int i = 0; i < n; i++) {
    for (int j = i + 1; j < n; j++) {
      if (compare(&array_bytes[j*size], &array_bytes[i*size]) < 0) {
        memcpy(tmp, &array_bytes[i*size], size);
        memcpy(&array_bytes[i*size], &array_bytes[j*size], size);
        memcpy(&array_bytes[j*size], tmp, size);
      }
    }
  }

  free(tmp);
}

int compare_int(const void *arg1, const void *arg2)
{
  const int i = *((int*) arg1);
  const int j = *((int*) arg2);

  if (i > j) {
    return 1;
  }
  else if (i < j) {
    return -1;
  }

  return 0;
}

int compare_lex(const void *arg1, const void *arg2)
{
  const char *str1 = (char*) arg1;
  const char *str2 = (char*) arg2;

  int i = 0;
  while (i < (int) strlen(str1) && i < (int) strlen(str2)) {
    const char l1 = tolower(str1[i]);
    const char l2 = tolower(str2[i]);

    if (l1 > l2) {
      return 1;
    }
    else if (l1 < l2) {
      return -1;
    }

    i++;
  }

  return 0;
}

int main(int argc, char **argv)
{
  srand(123);

  const int num_elements = atoi(argv[1]);

  int *x = malloc(num_elements*sizeof(int));

  // Randomize array and sort with specific algorithm for integers.
  for (int i = 0; i < num_elements; i++) {
    x[i] = rand();
  }

  printf("Sorting integer array with specific algorithm...\n");
  sort_integer(x, num_elements);

  // Randomize array and sort with generic algorithm.
  for (int i = 0; i < num_elements; i++) {
    x[i] = rand();
  }

  printf("Sorting integer array with generic algorithm...\n");
  sort(x, num_elements, sizeof(int), compare_int);

  // Create array of random character strings and sort with generic algorithm.
  str32 *strings = malloc(num_elements*sizeof(str32));
  for (int i = 0; i < num_elements; i++) {
    for (int j = 0; j < 31; j++) {
      // Select a random letter from 'a' (ASCII code 97) to 'z'.
      strings[i][j] = (char) 97 + rand()%26;
    }
    strings[i][31] = '\0';
  }

  printf("Sorting array of strings with generic algorithm...\n");
  sort(strings, num_elements, 32, compare_lex);

  free(x);
  free(strings);

  return 0;
}
