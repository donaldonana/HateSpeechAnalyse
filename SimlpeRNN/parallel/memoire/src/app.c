#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "utils.h"
#include "simplernn.h"
#include <time.h>
#include <unistd.h>
#include <string.h>
#include <pthread.h>
#include <sys/time.h>
// # define NUM_THREADS 2

struct timeval start_t , end_t ;
SimpleRNN *rnn;
Data *data ;
float lr;
int batch_size, NUM_THREADS, epoch  ;

pthread_mutex_t mutexRnn;

typedef struct thread_param thread_param;
struct thread_param{  
  SimpleRNN *rnn;
  DerivedSimpleRNN *grad;
  SimpleRNN *AVGgradient;
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
      batch_size = (unsigned long) atoi(argv[a + 1]);
      if ( batch_size == 0 ) {
        // usage(argv);
      }
    }  
    a += 1;

  }
}


void *ThreadTrain (void *params)  // Code du thread
{ 
  struct thread_param *mes_param ;
  mes_param = ( struct thread_param *) params ;
  mes_param->AVGgradient = malloc(sizeof(SimpleRNN));
  int nb_traite = 0;
	initialize_rnn_gradient(mes_param->rnn, mes_param->AVGgradient);
  for (int i = mes_param->start; i < mes_param->end; i++)
  {
    forward(mes_param->rnn, data->X[i], data->xcol , data->embedding);
    backforward(mes_param->rnn, data->xcol, data->Y[i], data->X[i], data->embedding, mes_param->grad, mes_param->AVGgradient);
    mes_param->loss = mes_param->loss + binary_loss_entropy(data->Y[i], mes_param->rnn->y);
    mes_param->acc = accuracy(mes_param->acc , data->Y[i], mes_param->rnn->y);

    nb_traite = nb_traite + 1; 
    if(nb_traite==batch_size || i == (mes_param->end -1))
    {	
      pthread_mutex_lock (&mutexRnn);
        gradient_descent(rnn, mes_param->AVGgradient,  nb_traite, lr);
        reinitialize_rnn(rnn, mes_param->rnn);
      pthread_mutex_unlock (&mutexRnn);
      nb_traite = 0;
    }
  }
  deallocate_rnn_derived(mes_param->rnn, mes_param->grad);
  deallocate_rnn(mes_param->rnn);
  deallocate_rnn_gradient(mes_param->rnn, mes_param->AVGgradient);
  pthread_exit (NULL);
}

int main(int argc, char **argv)
{
    // srand(time(NULL));
    pthread_mutex_init(&mutexRnn, NULL);
    data = malloc(sizeof(Data));
    get_data(data, 2);
    double totaltime;
    int n , r, end, start = 0 , size = 1000;
    int input = 128 , hidden = 64 , output = 2;
    float Loss , Acc ;
    void *status;

    // default parameters 
    lr = 0.01 ;
    epoch = 10;
    NUM_THREADS = 2;
    batch_size = 1;

    parse_input_args(argc, argv);
    thread_param *threads_params = malloc(sizeof(thread_param)*NUM_THREADS);
    pthread_t *threads = malloc(sizeof(pthread_t)*NUM_THREADS);
    pthread_attr_t attr ;

    /* Initialize and set thread detached attribute */
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

    rnn = malloc(sizeof(SimpleRNN));
    initialize_rnn(rnn, input, hidden, output);
    print_summary(rnn, epoch, batch_size, lr, NUM_THREADS);

                      printf("\n====== Training =======\n");

    gettimeofday(&start_t, NULL);
    n = size/NUM_THREADS;
    for (int e = 0; e < epoch; e++)
    {
        start = 0 ; 
        end = n-1 ;
        Loss = Acc = 0.0 ;
        printf("\nStart of epoch %d/%d \n", (e+1) , epoch);
            
        for ( int i=0; i < NUM_THREADS ; i ++) {
            threads_params[i].rnn = malloc(sizeof(SimpleRNN));
            copy_rnn(rnn, threads_params[i].rnn);
            threads_params[i].grad = malloc(sizeof(DerivedSimpleRNN));
            initialize_rnn_derived(threads_params[i].rnn , threads_params[i].grad);
            threads_params[i].loss = 0.0;
            threads_params[i].acc = 0.0;
            threads_params[i].start = start;
            threads_params[i].end = end;
            r = pthread_create (&threads[i] ,&attr ,ThreadTrain ,(void*)&threads_params[i]) ;
            if (r) {
              printf("ERROR; pthread_create() return code : %d\n", r);
              exit(-1);
            }
            start = end + 1;
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
          if (r) 
          {
            printf("ERROR; return code from pthread_join() is %d\n", r);
            exit(-1);
          }
          Loss = Loss + threads_params[t].loss ;
          Acc = Acc + threads_params[t].acc ;
          // printf("Main: thread completed %d an loss = %f and Accuracy = %f\n",t, (threads_params[t].loss)/n, (threads_params[t].acc)/n);
        }

        printf("--> Loss : %f  Accuracy : %f \n" , Loss/size, Acc/size);    
          
    }

    gettimeofday(&end_t, NULL);
    totaltime = (((end_t.tv_usec - start_t.tv_usec) / 1.0e6 + end_t.tv_sec - start_t.tv_sec) * 1000) / 1000;
    printf("\nTRAINING PHASE END IN %lf s\n" , totaltime);
    
    pthread_mutex_destroy(&mutexRnn);
    deallocate_rnn(rnn);
    free(threads);
    free(threads_params);
    pthread_exit(NULL);
}

