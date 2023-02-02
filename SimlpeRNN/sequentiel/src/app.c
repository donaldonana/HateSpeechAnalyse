#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "algebre.h"
#include "simplernn.h"
#include <time.h>
#include <string.h>
#include <pthread.h>


int main()
{
  // srand(time(NULL));
  Data* data = malloc(sizeof(Data));
  get_data(data);

  int epoch = 10;
  float lr = 0.01 ;
  int input = 128 , hidden = 64 , output = 2, mini_batch = 1;

  SimpleRNN *rnn = malloc(sizeof(SimpleRNN));
  SimpleRNN *AVGgradient = malloc(sizeof(SimpleRNN));
  gradient  *grad = malloc(sizeof(gradient));
  initialize_rnn(rnn, input, hidden, output);
  initialize_rnn(AVGgradient, input, hidden, output);
  initialize_rnn_derived(rnn , grad);

  print_summary(rnn, epoch, mini_batch, lr);
  training(epoch, rnn, grad, AVGgradient, data, 1000, mini_batch) ;

  // printf("\n ******************* TEST PHASE START *******************\n");
  // testing(rnn, data, datadim, embedding_matrix, train, target);

  return 0 ;
}
