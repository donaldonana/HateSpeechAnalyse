#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "utilities.h"
#include "lstm.h"
#include "layers.h"
#include <time.h>
#include <string.h>
#include <pthread.h>

#include "std_conf.h"


lstm_model_parameters params;

int main()
{

  memset(&params, 0, sizeof(params));
  params.learning_rate = 0.01;
  params.softmax_temp = SOFTMAX_TEMP;
  params.mini_batch_size = MINI_BATCH_SIZE;

  Data *data  = malloc(sizeof(Data));
  get_data(data);

  int X = data->ecol;
  int N = 64, Y = 2;
  int epoch = 15 ;

  lstm_rnn* lstm = e_calloc(1, sizeof(lstm_rnn));
  lstm_rnn* gradient = e_calloc(1, sizeof(lstm_rnn));
  lstm_rnn* AVGgradient = e_calloc(1, sizeof(lstm_rnn));
  lstm_values_cache **cache = alloc_cache_array(X, N, Y, data->xcol);

  lstm_init_model(X, N, Y , lstm, 0, &params); 
  lstm_init_model(X, N, Y , gradient , 1, &params);
  lstm_init_model(X, N, Y , AVGgradient , 1, &params);

  double *h_prev = get_zero_vector(lstm->N);
  double *c_prev = get_zero_vector(lstm->N);

  for (int e = 0; e < epoch ; e++)
  {
    printf("\nStart of epoch %d/%d \n", (e+1) , epoch);
    lstm_training(lstm, gradient, AVGgradient, data, cache, h_prev, c_prev);
  }
 
  lstm_free_model(lstm);
  lstm_free_model(gradient);
  lstm_free_model(AVGgradient);
  printf("\n initialization finish. \n");

}
