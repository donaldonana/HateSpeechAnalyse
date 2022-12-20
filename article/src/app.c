#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "utils.h"
#include "simplernn.h"
#include <time.h>
#include <string.h>
#include <pthread.h>

// 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
// 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

int main()
{
  // srand(time(NULL));
  Data *data  = malloc(sizeof(Data));
  get_data(data, 2);
  return 0 ;
}
