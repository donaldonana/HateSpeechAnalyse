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
  srand(time(NULL));
  printf("\n ***************** IMPORT PHASE START *****************\n");
  Data *data = malloc(sizeof(Data));
  get_data(data);
  
  //  **************** INITIALIZE THE RNN PHASE*****************
  SimpleRNN *rnn = malloc(sizeof(SimpleRNN));
  DerivedSimpleRNN *drnn = malloc(sizeof(DerivedSimpleRNN));
  int input = 128 , hidden = 64 , output = 2;
  initialize_rnn(rnn, input, hidden, output);
  initialize_rnn_derived(rnn , drnn);

  printf("\n ****************** TRAINING PHASE START ****************\n");
  training(30, rnn, drnn, data, 1000) ;

  // printf("\n ******************* TEST PHASE START *******************\n");
  // testing(rnn, data, datadim, embedding_matrix, train, target);

  return 0 ;
}
