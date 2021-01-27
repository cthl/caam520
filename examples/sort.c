#include <stdio.h>
#include <stdlib.h>
#include <string.h> // for memcpy()
#include <ctype.h> // for tolower()

// Try to understand the `sort_integer` funtion first!
void sort(void *array,
          int n,
          int size,
          int (*compare)(const void*, const void*))
{
  // We need to access individual elements of the array for sorting.
  // However, we cannot access them via array[i] or (array + i), because such
  // access is illegal for void pointers.
  // Hence, we overlay the array with a char array so that we can access the ith
  // byte as array_bytes[i].
  // (Note that char is not guaranteed to be exactly one byte, but on our
  // platform it is.)
  char *array_bytes = (char*) array;

  // Step 1: Create temporary storage for swapping elements.
  void *tmp = malloc(size);

  // Step 2: Loop over array as needed for selection sort.
  for (int i = 0; i < n; i++) {
    for (int j = i + 1; j < n; j++) {
      // Step 3: Compare elements.
      // Since we do not know what data we are dealing with, we must use the
      // comparison function provided by the user.
      if (compare(&array_bytes[j*size], &array_bytes[i*size]) < 0) {
        // Step 4: Swap elements.
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
  // Function accepts void pointers so it can be used in the `sort` function.
  // However, we need the integer values for our comparison.
  const int i = *((int*) arg1);
  const int j = *((int*) arg2);

  if (i > j) {
    return 1;
  }
  else if (i < j) {
    return -1;
  }

  // Both numbers are equal.
  return 0;
}

// Lexicographical ordering for strings.
int compare_lex(const void *arg1, const void *arg2)
{
  // Function accepts void pointers so it can be used in the `sort` function.
  // However, we need character strings for our comparison.
  const char *str1 = (char*) arg1;
  const char *str2 = (char*) arg2;

  int i = 0;
  while (i < (int) strlen(str1) && i < (int) strlen(str2)) {
    // Compare letters in a case insensitive way.
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

  // Strings are equal according to lexicographical ordering.
  return 0;
}

void sort_integer(int *array, int n)
{
  // Step 1: Create temporary storage for swapping elements.
  int tmp;

  // Step 2: Loop over array as needed for selection sort.
  for (int i = 0; i < n; i++) {
    for (int j = i + 1; j < n; j++) {
      // Step 3: Compare elements.
      if (array[j] < array[i]) {
        // Step 4: Swap elements.
        tmp = array[i];
        array[i] = array[j];
        array[j] = tmp;
      }
    }
  }
}

int main()
{
  // Create, sort, and print an array of integers.
  int x[6] = {9, 2, 5, 3, 7, 6};
  printf("before sorting:\n");
  for (int i = 0; i < 6; i++) {
    printf("x[%d] = %d\n", i, x[i]);
  }
  sort_integer(x, 6);
  printf("after sorting:\n");
  for (int i = 0; i < 6; i++) {
    printf("x[%d] = %d\n", i, x[i]);
  }

  // Create, sort, and print an array of integers, this time using the generic
  // sorting algorithm.
  int y[6] = {9, 2, 5, 3, 7, 6};
  printf("before sorting:\n");
  for (int i = 0; i < 6; i++) {
    printf("y[%d] = %d\n", i, y[i]);
  }
  sort(y, 6, sizeof(int), compare_int);
  printf("after sorting:\n");
  for (int i = 0; i < 6; i++) {
    printf("y[%d] = %d\n", i, y[i]);
  }

  // Create, sort, and print an array of character strings using the same
  // generic sorting algorithm as before.
  char names[6][32] = {"Maria",
                       "Bob",
                       "Alice",
                       "Robert Staunton",
                       "Jules Verne",
                       "Robert Smith"};

  printf("before sorting:\n");
  for (int i = 0; i < 6; i++) {
    printf("names[%d] = %s\n", i, names[i]);
  }
  sort(names, 6, 32, compare_lex);
  printf("after sorting:\n");
  for (int i = 0; i < 6; i++) {
    printf("names[%d] = %s\n", i, names[i]);
  }

  return 0;
}
