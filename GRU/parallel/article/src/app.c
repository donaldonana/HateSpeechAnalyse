#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "utilities.h"
#include "gru.h"
#include <time.h>
#include <unistd.h>
#include <string.h>
#include <pthread.h>
#include <sys/time.h>
// # define NUM_THREADS 2

struct timeval start_t , end_t ;
gru_rnn *gru;
Data *data ;
float lr;
int MINI_BATCH_SIZE, NUM_THREADS, epoch  ;

pthread_mutex_t mutexRnn;

typedef struct thread_param thread_param;
struct thread_param{  
  gru_rnn* gru;
  gru_rnn* gradient  ;
  gru_rnn* AVGgradient ;
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
    }  
    a += 1;

  }
}



void *ThreadTrain (void *params) // Code du thread
{ 
  struct thread_param *mes_param ;
  int nb_traite = 0;
  mes_param = ( struct thread_param *) params ;
  mes_param->AVGgradient = e_calloc(1, sizeof(gru_rnn));
  gru_init_model(gru->X, gru->N, gru->Y , mes_param->AVGgradient , 1);
   
  for (int i = mes_param->start; i < mes_param->end; i++)
  {
    // forward
    gru_forward(mes_param->gru, data->X[i], mes_param->gru->cache, data);
    // compute loss
    mes_param->loss = mes_param->loss + binary_loss_entropy(data->Y[i], mes_param->gru->probs, data->ycol);
    // compute accuracy training
    mes_param->acc = accuracy(mes_param->acc , data->Y[i],  mes_param->gru->probs, data->ycol);
    // backforward
    gru_backforward(mes_param->gru, data->Y[i], (data->xcol-1), mes_param->gru->cache, mes_param->gradient);
    sum_gradients(mes_param->AVGgradient, mes_param->gradient);
    nb_traite = nb_traite + 1; 

     if(nb_traite==MINI_BATCH_SIZE || i == (mes_param->end -1))
    {	
      pthread_mutex_lock (&mutexRnn);
        gradients_decend(mes_param->gru, mes_param->AVGgradient, lr, nb_traite);
      pthread_mutex_unlock (&mutexRnn);
      nb_traite = 0;
    }

    gru_zero_the_model(mes_param->gradient);
    set_vector_zero(gru->h_prev, gru->N);

  }
  gru_free_model(mes_param->gradient);
  gru_free_model(mes_param->AVGgradient);

  pthread_exit (NULL);
}

int main(int argc, char **argv)
{
    // srand(time(NULL));
    pthread_mutex_init(&mutexRnn, NULL);
    data = malloc(sizeof(Data));
    get_data(data);
    double totaltime;
    void *status;
    // char filaname[] = "gru.json";
    int n , r, end, start = 0 , size = 4460;
    int X = data->ecol , N = 64, Y = 2;
    float Loss , Acc ;
    
    // default parameters 
    lr = 0.01 ;
    epoch = 20;
    NUM_THREADS = 2;
    MINI_BATCH_SIZE = 1;

    parse_input_args(argc, argv);
    thread_param *threads_params = malloc(sizeof(thread_param)*NUM_THREADS);
    pthread_t *threads = malloc(sizeof(pthread_t)*NUM_THREADS);
    pthread_attr_t attr ;

    /* Initialize and set thread detached attribute */
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

    gru = e_calloc(1, sizeof(gru_rnn));
    gru_init_model(X, N, Y , gru, 0); 
    gru_rnn* AVGgradient = e_calloc(1, sizeof(gru_rnn));
    gru_init_model(X, N, Y , AVGgradient, 1); 

    print_summary(gru, epoch, MINI_BATCH_SIZE, lr, NUM_THREADS);

                printf("\n====== Training =======\n");

    gettimeofday(&start_t, NULL);
    n = size/NUM_THREADS;
    for (int e = 0; e < epoch; e++)
    {
        start = 0 ; 
        end = n ;
        Loss = Acc = 0.0 ;
        printf("\nStart of epoch %d/%d \n", (e+1) , epoch);
        for ( int i=0; i < NUM_THREADS ; i ++) 
        {
            threads_params[i].gru = e_calloc(1, sizeof(gru_rnn));
            gru_init_model(X, N, Y , threads_params[i].gru , 0);
            copy_gru(gru, threads_params[i].gru);
            threads_params[i].gradient = e_calloc(1, sizeof(gru_rnn));
            gru_init_model(X, N, Y , threads_params[i].gradient , 1);
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
        for(int t=0; t<NUM_THREADS; t++) {
          r = pthread_join(threads[t], &status);
          if (r) {
            printf("ERROR; return code from pthread_join() is %d\n", r);
            exit(-1);
          }
          somme_gradient(AVGgradient, threads_params[t].gru);
          Loss = Loss + threads_params[t].loss ;
          Acc  = Acc + threads_params[t].acc ;

        }

        modelUpdate(gru, AVGgradient, NUM_THREADS);
        printf("--> Loss : %f  Accuracy : %f \n" , Loss/size, Acc/size);  

        // lstm_store_net_layers_as_json(gru, filaname); 
          
    }

    gettimeofday(&end_t, NULL);
    totaltime = (((end_t.tv_usec - start_t.tv_usec) / 1.0e6 + end_t.tv_sec - start_t.tv_sec) * 1000) / 1000;
    printf("\nTRAINING PHASE END IN %lf s\n" , totaltime);
    
    pthread_mutex_destroy(&mutexRnn);
    gru_free_model(gru);
    free(threads);
    free(threads_params);
    pthread_exit(NULL);
}

