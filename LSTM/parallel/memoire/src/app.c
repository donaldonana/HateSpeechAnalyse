#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "utilities.h"
#include "lstm.h"
#include "std_conf.h"
#include <time.h>
#include <unistd.h>
#include <string.h>
#include <pthread.h>
#include <sys/time.h>
// # define NUM_THREADS 2

struct timeval start_t , end_t ;
pthread_mutex_t mutexRnn;
lstm_rnn *lstm;
Data *data ;
float lr , VALIDATION_SIZE ;
int epoch, MINI_BATCH_SIZE, NUM_THREADS , HIDEN_SIZE;

pthread_mutex_t mutexRnn;

typedef struct thread_param thread_param;
struct thread_param{  
  lstm_rnn* lstm;
  lstm_rnn* gradient  ;
  lstm_rnn* AVGgradient ;
  int start;
  int end;
  float loss;
  float acc;
};


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
    } else if ( !strcmp(argv[a], "-validation") ) {
      VALIDATION_SIZE =  atof(argv[a + 1]);
      if ( VALIDATION_SIZE < 0.1 || VALIDATION_SIZE > 0.3) {
        // usage(argv);
        VALIDATION_SIZE = 0.1;
      }
      
    } else if ( !strcmp(argv[a], "-hiden") ) {
      HIDEN_SIZE =  atoi(argv[a + 1]);
      if ( HIDEN_SIZE <= 2 || HIDEN_SIZE > 500) {
        // usage(argv);
        HIDEN_SIZE = 16;
      }
      
    }
    a += 1;

  }
}

void shuffle(int *array, size_t n)
{
    if (n > 1) 
    {
        size_t i;
        for (i = 0; i < n - 1; i++) 
        {
          size_t j = i + rand() / (RAND_MAX / (n - i) + 1);
          int t = array[j];
          array[j] = array[i];
          array[i] = t;
        }
    }
}

void *ThreadTrain (void *params) // Code du thread
{ 
  struct thread_param *mes_param ;
  mes_param = ( struct thread_param *) params ;
  mes_param->AVGgradient = e_calloc(1, sizeof(lstm_rnn));
  lstm_init_model(lstm->X, lstm->N, lstm->Y , mes_param->AVGgradient , 1);
  int nb_traite = 0 , k = 0, j = 0, n =  (mes_param->end - mes_param->start + 1) ;

  int *TrainIdx = malloc((n)*sizeof(int));
  for (int i = mes_param->start; i < mes_param->end ; i++)
  {
    TrainIdx[j] = i ; 
    j = j + 1;
  }
  shuffle(TrainIdx,(n-1));
  j = 0;
   
  for (int i = mes_param->start; i < mes_param->end; i++)
  {
    k = TrainIdx[j];
    // Forward
    lstm_forward(mes_param->lstm, data->X[k], mes_param->lstm->cache, data);
    // Compute loss
    mes_param->loss = mes_param->loss + binary_loss_entropy(data->Y[k], mes_param->lstm->probs, data->ycol);
    // Compute accuracy training
    mes_param->acc = accuracy(mes_param->acc , data->Y[k],  mes_param->lstm->probs, data->ycol);
    // Backforward
    lstm_backforward(mes_param->lstm, data->Y[k], (data->xcol-1), mes_param->lstm->cache, mes_param->gradient);
    sum_gradients(mes_param->AVGgradient, mes_param->gradient);
    nb_traite = nb_traite + 1; 
    // Update The Central LSTM
    if(nb_traite==MINI_BATCH_SIZE || i == (mes_param->end -1))
    {	
      pthread_mutex_lock (&mutexRnn);
        gradients_decend(lstm, mes_param->AVGgradient, lr, nb_traite);
        copy_lstm(lstm, mes_param->lstm);
        nb_traite = 0;
      pthread_mutex_unlock (&mutexRnn);
    }
    lstm_zero_the_model(mes_param->gradient);
    set_vector_zero(lstm->h_prev, lstm->N);
    set_vector_zero(lstm->c_prev, lstm->N);
    j = j + 1;
  }
  lstm_free_model(mes_param->gradient);
  lstm_free_model(mes_param->AVGgradient);
  free(TrainIdx);
  pthread_exit (NULL);
}

int main(int argc, char **argv)
{
    srand(time(NULL));
    pthread_mutex_init(&mutexRnn, NULL);
    FILE *fl  = fopen(LOSS_FILE_NAME, "w");
    FILE *fa  = fopen(ACC_FILE_NAME,  "w");
    FILE *fv  = fopen(VAL_LOSS_FILE_NAME,  "w");
    data = malloc(sizeof(Data));
    double totaltime;
    void *status;
    int n , r, end, start = 0 , size , stop = 0, e = 0 , X , N , Y;
    float Loss , Acc , val_loss, best_loss = 100 ;
    
    // Set Parameters And Retreive data
    lr = 0.01; epoch = 20; NUM_THREADS = 2; MINI_BATCH_SIZE = 16; VALIDATION_SIZE = 0; HIDEN_SIZE = 64; 
    parse_input_args(argc, argv);
    get_split_data(data, VALIDATION_SIZE);
    size = (data->start_val - 1) ; X = data->ecol ; Y = data->ycol; N = HIDEN_SIZE; n = size/NUM_THREADS; 

    /* Initialize and Set thread params */
    thread_param *threads_params = malloc(sizeof(thread_param)*NUM_THREADS);
    pthread_t    *threads = malloc(sizeof(pthread_t)*NUM_THREADS);
    pthread_attr_t attr ;

    /* Initialize and set thread detached attribute */
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

    /* Initialize central Model */
    lstm = e_calloc(1, sizeof(lstm_rnn));
    lstm_init_model(X, N, Y , lstm, 0); 
    print_summary(lstm, epoch, MINI_BATCH_SIZE, lr, NUM_THREADS);

    printf("\n====== Training =======\n");

    gettimeofday(&start_t, NULL);
    while (e < epoch )
    {
    	srand(time(NULL));

      start = 0 ; 
      end = n ;
      Loss = Acc = 0.0 ;
      printf("\nStart of epoch %d/%d \n", (e+1) , epoch);
      /* Create And Start The Threads */
      for ( int i=0; i < NUM_THREADS ; i ++) 
      {
        threads_params[i].lstm = e_calloc(1, sizeof(lstm_rnn));
        lstm_init_model(X, N, Y , threads_params[i].lstm , 0);
        copy_lstm(lstm, threads_params[i].lstm);
        threads_params[i].gradient = e_calloc(1, sizeof(lstm_rnn));
        lstm_init_model(X, N, Y , threads_params[i].gradient , 1);
        threads_params[i].loss = 0.0;
        threads_params[i].acc = 0.0;
        threads_params[i].start = start;
        threads_params[i].end = end;
        r = pthread_create (&threads[i] ,&attr ,ThreadTrain ,(void*)&threads_params[i]) ;
        if (r) {
            printf("ERROR; pthread_create() return code : %d\n", r);
            exit(-1);
        }
        start = end ;
        end = end + n;
        if (i == (NUM_THREADS-1) )
        {
          end = end + size%NUM_THREADS ;
        }

      }
      /* Free attribute and wait for the other threads */
      pthread_attr_destroy(&attr);
      for(int t=0; t<NUM_THREADS; t++) 
      {
        r = pthread_join(threads[t], &status);
        if (r) {
        printf("ERROR; return code from pthread_join() is %d\n", r);
          exit(-1);
        }
        Loss = Loss + threads_params[t].loss ;
        Acc  = Acc + threads_params[t].acc ;
      }
      printf("--> Train Loss : %f || Train Accuracy : %f \n" , Loss/size, Acc/size); 
      fprintf(fl,"%d,%.6f\n", e+1 , Loss/size);
      fprintf(fa,"%d,%.6f\n", e+1 , Acc/size);
      /* Validation Phase And Early Stoping */
      val_loss = lstm_validation(lstm, data);
      fprintf(fv,"%d,%.6f\n", e+1 , val_loss);
      if (val_loss < best_loss)
      {
        printf("\nsave");
        lstm_store_net_layers_as_json(lstm, MODEL_FILE_NAME); 
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
    pthread_mutex_destroy(&mutexRnn);
    lstm_free_model(lstm);
    free(threads);
    free(threads_params);
    pthread_exit(NULL);

}

