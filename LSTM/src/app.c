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

#define ITERATIONS  100000000
#define NO_EPOCHS   0 

 lstm_model_parameters_t params;

int main()
{

  memset(&params, 0, sizeof(params));

  params.iterations = ITERATIONS;
  params.epochs = NO_EPOCHS;
  params.loss_moving_avg = LOSS_MOVING_AVG;
  params.learning_rate = STD_LEARNING_RATE;
  params.momentum = STD_MOMENTUM;
  params.lambda = STD_LAMBDA;
  params.softmax_temp = SOFTMAX_TEMP;
  params.mini_batch_size = MINI_BATCH_SIZE;
  params.gradient_clip_limit = GRADIENT_CLIP_LIMIT;
  params.learning_rate_decrease = STD_LEARNING_RATE_DECREASE;
  params.stateful = STATEFUL;
  params.beta1 = 0.9;
  params.beta2 = 0.999;
  params.gradient_fit = GRADIENTS_FIT;
  params.gradient_clip = GRADIENTS_CLIP;
  params.decrease_lr = DECREASE_LR;
  params.model_regularize = MODEL_REGULARIZE;
  params.optimizer = OPTIMIZE_ADAM;


  int X = 2;
  int N = 64;
  int Y = 2; 
  double tab[2] = {0,1};
  lstm_model_t* lstm = e_calloc(1, sizeof(lstm_model_t));
  lstm_init_model(X, N, Y , lstm, 0, &params); 

  lstm_values_cache_t **cache_layers = e_calloc(X, sizeof(lstm_values_cache_t));

  double *h_prev = get_zero_vector(N);
  double *c_prev = get_zero_vector(N);

  for (int t = 0; t < X; t++)
  {
    cache_layers[t] = lstm_cache_container_init(X, N, Y);
    lstm_forward_propagate(lstm, tab, h_prev, c_prev, cache_layers[t]);
    copy_vector(h_prev, cache_layers[t]->h, N);
    copy_vector(c_prev, cache_layers[t]->c, N);

    for (int i = 0; i < Y; i++)
    {
      printf("\n %lf " , cache_layers[t]->probs[i]);
      printf(" , %lf " , cache_layers[t]->probs[i]);

    }
    

  }
  
  
  lstm_free_model(lstm);

  printf("\n initialization finish. \n");
   
}
