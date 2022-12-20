#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "utils.h"
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
int batch_size  ;

pthread_mutex_t mutexRnn;

typedef struct thread_param thread_param;
struct thread_param{  
SimpleRNN *rnn;
DerivedSimpleRNN *drnn;
dSimpleRNN *grnn;
int start;
int end;
float loss;
float acc;
};


// struct thread_param threads_params[NUM_THREADS];

void *ThreadTrain (void *params) { // Code du thread
    struct thread_param *mes_param ;
    mes_param = ( struct thread_param *) params ;
    mes_param->grnn = malloc(sizeof(dSimpleRNN));
    int nb_traite = 0;
	initialize_rnn_gradient(mes_param->rnn, mes_param->grnn);
    for (int i = mes_param->start; i < mes_param->end; i++)
    {
        forward(mes_param->rnn, data->X[i], data->xcol , data->embedding);
        backforward(mes_param->rnn, data->xcol, data->Y[i], data->X[i], data->embedding, 
        mes_param->drnn, mes_param->grnn);
        mes_param->loss = mes_param->loss + binary_loss_entropy(data->Y[i], mes_param->rnn->y);
        mes_param->acc = accuracy(mes_param->acc , data->Y[i], mes_param->rnn->y);
        
        if(nb_traite==batch_size || i == (mes_param->end -1))
        {	
        pthread_mutex_lock (&mutexRnn);
            gradient_descent(rnn, mes_param->grnn,  nb_traite, lr);
            reinitialize_rnn(rnn, mes_param->rnn);
        pthread_mutex_unlock (&mutexRnn);
            nb_traite = 0;
        }
        nb_traite = nb_traite + 1;
        
    }
    deallocate_rnn_derived(mes_param->rnn, mes_param->drnn);
    deallocate_rnn(mes_param->rnn);
    deallocate_rnn_gradient(mes_param->rnn, mes_param->grnn);
    pthread_exit (NULL) ;
}

int main(int argc, char **argv)
{
    // srand(time(NULL));
    pthread_mutex_init(&mutexRnn, NULL);
    lr = 0.01 ;
    // float preLog ;


    double totaltime;
    data = malloc(sizeof(Data));
    get_data(data, 2);
    int size = 2000, divide = 0;
    int epoch = 20, NUM_THREADS = 2;
    batch_size = 16;
    int n , start , end;
    float Loss , Acc ;

    if(argc >= 4)
    {
	    NUM_THREADS = atoi(argv[1]);
	    epoch    = atoi(argv[2]);
        batch_size = atoi(argv[3]);
    }
    n = size/NUM_THREADS;
    start = 0 ; end = n-1;
    thread_param *threads_params = malloc(sizeof(thread_param)*NUM_THREADS);
    int input = 128 , hidden = 64 , output = 2;
    pthread_t *threads = malloc(sizeof(pthread_t)*NUM_THREADS);
    pthread_attr_t attr ;
    void *status;
    int r;

    printf("\n----Thread create phase start---- \n");
    /* Initialize and set thread detached attribute */
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

    gettimeofday(&start_t, NULL);
    printf("\n %d \n", n);

    rnn = malloc(sizeof(SimpleRNN));
    initialize_rnn(rnn, input, hidden, output);
    // preLog = testing(rnn, data, data->start_val, (data->end_val-2000));

    
    for (int i = 0; i < epoch; i++)
    {
        start = 0 ; 
        end = n-1 ;
        Loss = Acc = 0.0 ;
        printf("\n epoch %d \n", (i+1));
            
        for ( int i=0; i < NUM_THREADS ; i ++) {
            threads_params[i].rnn = malloc(sizeof(SimpleRNN));
            copy_rnn(rnn, threads_params[i].rnn);
            threads_params[i].drnn = malloc(sizeof(DerivedSimpleRNN));
            initialize_rnn_derived(threads_params[i].rnn , threads_params[i].drnn);
            threads_params[i].loss = 0.0;
            threads_params[i].acc = 0.0;
            threads_params[i].start = start;
            threads_params[i].end = end;
            r = pthread_create (&threads[i] ,&attr ,ThreadTrain ,(void*)&threads_params[i]) ;
            if (r) {
                printf("ERROR; pthread_create() return code : %d\n", r);
                exit(-1);
            }
            printf("Thread %d has starded \n", i);
            start = end + 1;
            end = end + n;
        }
        printf("----Thread create phase end----\n");


        /* Free attribute and wait for the other threads */
        pthread_attr_destroy(&attr);
        for(int t=0; t<NUM_THREADS; t++) {
            r = pthread_join(threads[t], &status);
            if (r) {
                printf("ERROR; return code from pthread_join() is %d\n", r);
                exit(-1);
            }
            Loss = Loss + threads_params[t].loss ;
            Acc = Acc + threads_params[t].acc ;
            printf("Main: thread completed %d an loss = %f and Accuracy = %f\n",t, 
            (threads_params[t].loss)/n, (threads_params[t].acc)/n);
        }

        // printf("--> Loss : %f  \n" , Loss/size);    
        printf("--> Loss : %f  Accuracy : %f \n" , Loss/size, Acc/size);    
        // val_loss = testing(rnn, data, data->start_val, (data->end_val-2000));
        // printf("--> Loss : %f  \n" , val_loss);    


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
