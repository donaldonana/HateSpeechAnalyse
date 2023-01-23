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

 lstm_model_parameters params;

int main()
{

  memset(&params, 0, sizeof(params));

  params.iterations = ITERATIONS;
  params.epochs = NO_EPOCHS;
  params.loss_moving_avg = LOSS_MOVING_AVG;
  params.learning_rate = 0.01;
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



  Data *data  = malloc(sizeof(Data));
  get_data(data);

  int X = data->ecol;
  int N = 64, Y = 2, t;
  int epoch = 15 ;
  float Loss;

  lstm_rnn* lstm = e_calloc(1, sizeof(lstm_rnn));
  lstm_rnn* gradient = e_calloc(1, sizeof(lstm_rnn) );
  lstm_init_model(X, N, Y , lstm, 0, &params); 
  lstm_init_model(X, N, Y , gradient , 1, &params);

  lstm_values_cache **cache = malloc((data->xcol+1)*sizeof(lstm_values_cache));
  for (t = 0; t < X; t++)
  {
    cache[t] = lstm_cache_container_init(X, N, Y);
  }
  double *h_prev = get_zero_vector(lstm->N);
  double *c_prev = get_zero_vector(lstm->N);


  
  for (int e = 0; e < epoch ; e++)
  {

    Loss = 0.0;
    printf("\nStart of epoch %d/%d \n", (e+1) , epoch);
     
    for (int i = 0; i < 1000; i++)
    {
      // forward
      lstm_forward(lstm, data->X[i], h_prev, c_prev, cache, data);
      Loss = Loss + binary_loss_entropy(data->Y[i], lstm->probs);

      // backforward
      lstm_backforward(lstm, data->Y[i], (data->xcol-1), cache, gradient);
      // update
      gradients_decend(lstm, gradient);

      lstm_zero_the_model(gradient);
      set_vector_zero(h_prev, N);
      set_vector_zero(c_prev, N);
    }
    Loss = Loss/1000;
    printf("%lf \n" , Loss);    


  
  }

  // for (int  i = 0; i < N; i++)
  // {
  //   printf("%lf \n" , gradient->bf[i]);    

  // }
  

  // for (int i = 0; i < Y*N; i++)
  // {
  //   printf("\n %d ---%lf ", i, gradient->Wy[i]);
  // }
  // printf("\n +++++ %lf +++++", cache_layers[0]->h[0]);
  

  lstm_free_model(lstm);
  printf("\n initialization finish. \n");
}
