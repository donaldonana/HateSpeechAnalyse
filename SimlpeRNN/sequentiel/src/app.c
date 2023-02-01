#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "utils.h"
#include "simplernn.h"
#include <time.h>
#include <string.h>
#include <pthread.h>


int main()
{
  srand(time(NULL));
  Data *data = malloc(sizeof(Data));
  get_data(data);
  int mini_batch = 32;
  int epoch = 20;
  
  SimpleRNN *rnn = malloc(sizeof(SimpleRNN));
  DerivedSimpleRNN *drnn = malloc(sizeof(DerivedSimpleRNN));
  dSimpleRNN *grnn = malloc(sizeof(dSimpleRNN));
  int input = 128 , hidden = 64 , output = 2;
  initialize_rnn(rnn, input, hidden, output);
  initialize_rnn_derived(rnn , drnn);
  initialize_rnn_gradient(rnn, grnn);

  training(20, rnn, drnn, grnn, data, 10000, mini_batch) ;

  // printf("\n ******************* TEST PHASE START *******************\n");
  // testing(rnn, data, datadim, embedding_matrix, train, target);

  return 0 ;
}
