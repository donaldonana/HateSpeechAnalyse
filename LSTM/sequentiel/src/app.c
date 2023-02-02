#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "utilities.h"
#include "lstm.h"
#include "layers.h"
#include <time.h>
#include <string.h>
#include <pthread.h>




float lr;
int MINI_BATCH_SIZE, NUM_THREADS, epoch  ;


void parse_input_args(int argc, char** argv)
{
  int a = 0;

  while ( a < argc ) {

    if ( argc <= (a+1) ){ 
      break; // All flags have values attributed to them
    }

    if ( !strcmp(argv[a], "-lr") ) {
      lr = atof(argv[a + 1]);
      if ( lr == 0.0 ) {
        // usage(argv);
      }
    } else if ( !strcmp(argv[a], "-thread") ) {
      NUM_THREADS = atoi(argv[a + 1]);
      if ( NUM_THREADS <= 0 ) {
        // usage(argv);
      }
    } else if ( !strcmp(argv[a], "-epoch") ) {
      epoch = (unsigned long) atoi(argv[a + 1]);
      if ( epoch == 0 ) {
        // usage(argv);
      }
    } else if ( !strcmp(argv[a], "-batch") ) {
      MINI_BATCH_SIZE = (unsigned long) atoi(argv[a + 1]);
      if ( MINI_BATCH_SIZE == 0 ) {
        // usage(argv);
      }
    }  
    a += 1;

  }
}



int main()
{

  Data *data  = malloc(sizeof(Data));
  get_data(data);
  lr = 0.01;
  MINI_BATCH_SIZE = 1;
  int X = data->ecol;
  int N = 64, Y = 2;
  epoch = 15 ;
  
  lstm_rnn* lstm = e_calloc(1, sizeof(lstm_rnn));
  lstm_rnn* gradient = e_calloc(1, sizeof(lstm_rnn));
  lstm_rnn* AVGgradient = e_calloc(1, sizeof(lstm_rnn));
  lstm_values_cache **cache = alloc_cache_array(X, N, Y, data->xcol);
  lstm_init_model(X, N, Y , lstm, 0); 
  lstm_init_model(X, N, Y , gradient , 1);
  lstm_init_model(X, N, Y , AVGgradient , 1);

  double *h_prev = get_zero_vector(lstm->N);
  double *c_prev = get_zero_vector(lstm->N);

  print_summary(lstm, epoch, MINI_BATCH_SIZE, lr, NUM_THREADS);
  printf("\n====== Training =======\n");
  for (int e = 0; e < epoch ; e++)
  {
    printf("\nStart of epoch %d/%d \n", (e+1) , epoch);
    lstm_training(lstm, gradient, AVGgradient, MINI_BATCH_SIZE, lr, data, cache, h_prev, c_prev);
  }
 
  lstm_free_model(lstm);
  lstm_free_model(gradient);
  lstm_free_model(AVGgradient);
  printf("\n initialization finish. \n");

}
