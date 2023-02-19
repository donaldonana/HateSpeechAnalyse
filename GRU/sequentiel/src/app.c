#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "utilities.h"
#include "gru.h"
#include "layers.h"
#include <time.h>
#include <string.h>
#include <sys/time.h>
#include <pthread.h>

struct timeval start_t , end_t ;

float lr;
int MINI_BATCH_SIZE, epoch  ;

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



int main(int argc, char **argv)
{

  Data *data  = malloc(sizeof(Data));
  get_data(data);

  lr = 0.01;
  double totaltime;
  MINI_BATCH_SIZE = 1;
  int X = data->ecol, N = 64, Y = 2;
  epoch = 15 ;
  
  parse_input_args(argc, argv);
  gru_rnn* gru = e_calloc(1, sizeof(gru_rnn));
  gru_rnn* gradient = e_calloc(1, sizeof(gru_rnn));
  gru_rnn* AVGgradient = e_calloc(1, sizeof(gru_rnn));
  gru_init_model(X, N, Y , gru, 0); 
  gru_init_model(X, N, Y , gradient , 1);
  gru_init_model(X, N, Y , AVGgradient , 1);
  print_summary(gru, epoch, MINI_BATCH_SIZE, lr);

    printf("\n====== Training =======\n");

  gettimeofday(&start_t, NULL);
  for (int e = 0; e < epoch ; e++)
  {
    printf("\nStart of epoch %d/%d \n", (e+1) , epoch);
    gru_training(gru, gradient, AVGgradient, MINI_BATCH_SIZE, lr, data);
  }

  gettimeofday(&end_t, NULL);
  totaltime = (((end_t.tv_usec - start_t.tv_usec) / 1.0e6 + end_t.tv_sec - start_t.tv_sec) * 1000) / 1000;
  printf("\nTRAINING PHASE END IN %lf s\n" , totaltime);
    
  gru_free_model(gru);
  gru_free_model(gradient);
  gru_free_model(AVGgradient);
  printf("\n initialization finish. \n");

}