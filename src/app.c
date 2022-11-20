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
  float **embedding_matrix = NULL;
  int **data = NULL;
  int *target = NULL;
  int train ;
  int datadim[2];
  int embeddim[2];
  data = GetData(datadim );
  embedding_matrix = GetEmbedding(embeddim);
  target = load_target(target);
  train = (int) datadim[0] * 0.7 ; 
  printf("\n train , from 0 to %d \n" , train);
  printf("\n test ,  from %d to %d \n" , train+1 , datadim[0]-1);


  //  **************** INITIALIZE THE RNN PHASE*****************
  SimpleRNN *rnn = malloc(sizeof(SimpleRNN));
  DerivedSimpleRNN *drnn = malloc(sizeof(DerivedSimpleRNN));
  int input = 128 , hidden = 64 , output = 2;
  initialize_rnn(rnn, input, hidden, output);
  initialize_rnn_derived(rnn , drnn);

  printf("\n ****************** TRAINING PHASE START ****************\n");
  training(40, data, datadim, embedding_matrix, target, rnn, drnn, 2000) ;

  printf("\n ******************* TEST PHASE START *******************\n");
  testing(rnn, data, datadim, embedding_matrix, train, target);

  //************************ FREE MEMORY PHASE **********************
  deallocate_dynamic_float_matrix(embedding_matrix, embeddim[0]);
  deallocate_dynamic_int_matrix(data, datadim[0]);
  free(target);


  return 0 ;
}
