#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "utilities.h"
#include "simplernn.h"
#include "std_conf.h"
#include "layers.h"
#include <time.h>
#include <string.h>
#include <sys/time.h>

#include <pthread.h>

struct timeval start_t , end_t ;
float lr, VALIDATION_SIZE ; 
int MINI_BATCH_SIZE, epoch , HIDEN_SIZE;

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
      else if ( !strcmp(argv[a], "-validation") ) {
      VALIDATION_SIZE =  atof(argv[a + 1]);
      if ( VALIDATION_SIZE < 0.1 || VALIDATION_SIZE > 0.3) {
        // usage(argv);
        VALIDATION_SIZE = 0.1;
      }
      
    }
      else if ( !strcmp(argv[a], "-hiden") ) {
      HIDEN_SIZE =  atoi(argv[a + 1]);
      if ( HIDEN_SIZE < 4 || HIDEN_SIZE > 500) {
        // usage(argv);
        HIDEN_SIZE = 16;
      }
      
    }
    a += 1;

  }
}


int main(int argc, char **argv)
{
  FILE *fl  = fopen(LOSS_FILE_NAME, "w");
  FILE *fa  = fopen(ACC_FILE_NAME,  "w");
  FILE *fv  = fopen(VAL_LOSS_FILE_NAME,  "w"); 
  FILE *ft  = fopen(TEST_FILE_NAME,  "w"); 

  Data *data  = malloc(sizeof(Data));
  double totaltime;
  float val_loss, best_loss = 100 ;
  int X, Y, N, stop = 0, e = 0 ; 
  lr = 0.01; MINI_BATCH_SIZE = 16; epoch = 10 ; HIDEN_SIZE = 64 ;
  parse_input_args(argc, argv);
  get_split_data(data, VALIDATION_SIZE);
  Y = data->ycol; X = data->ecol; N = HIDEN_SIZE ;
  
  SimpleRnn* rnn = e_calloc(1, sizeof(SimpleRnn));
  SimpleRnn* gradient = e_calloc(1, sizeof(SimpleRnn));
  SimpleRnn* AVGgradient = e_calloc(1, sizeof(SimpleRnn));
  rnn_init_model(X, N, Y , rnn, 0); 
  rnn_init_model(X, N, Y , gradient , 1);
  rnn_init_model(X, N, Y , AVGgradient , 1);
  print_summary(rnn, epoch, MINI_BATCH_SIZE, lr);

    printf("\n====== Training =======\n");
    
  gettimeofday(&start_t, NULL);

  while (e < epoch && stop < 4)
  {
    printf("\nStart of epoch %d/%d \n", (e+1) , epoch);
    // Training 
    rnn_training(rnn, gradient, AVGgradient, MINI_BATCH_SIZE, lr, data, e+1, fl , fa);
    // Validation And Early Stoping
    val_loss = rnn_validation(rnn, data);
    fprintf(fv,"%d,%.6f\n", e+1 , val_loss);
    if (val_loss < 0.9*best_loss)
    {
      printf("\nsave");
      rnn_store_net_layers_as_json(rnn, MODEL_FILE_NAME); 
      stop = 0;
      best_loss = val_loss;
    }
    else
    {
      stop = stop + 1;
    }
    e = e + 1 ; 
  }

  
  gettimeofday(&end_t, NULL);
  totaltime = (((end_t.tv_usec - start_t.tv_usec) / 1.0e6 + end_t.tv_sec - start_t.tv_sec) * 1000) / 1000;
  printf("\nTRAINING PHASE END IN %lf s\n" , totaltime);

  printf("\n====== Test Phase ======\n");
  printf(" \n...\n");
  rnn_test(rnn, data, ft);
  printf("\n");


  // Free section.
  rnn_free_model(rnn);
  rnn_free_model(gradient);
  rnn_free_model(AVGgradient);
  fclose(fl);
  fclose(fv);
  fclose(fa);

}
