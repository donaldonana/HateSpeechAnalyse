#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "utils.h"
#include "simplernn.h"
#include <time.h>
#include <string.h>
#include <pthread.h>

// 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

int main()
{
    srand(time(NULL));
    double time;
    clock_t start, end ;


    // ***************** IMPORT AND SPLIT IT INTO DATA TRAIN AND DATA TEST *****************
    
    printf("\n ************************ IMPORT PHASE START ************************ \n");
    float **embedding_matrix = NULL;
    int **data = NULL;
    int *target = NULL;
    int train ;
    int embeddim[2];
    int datadim[2];
    data = GetData(datadim );
    embedding_matrix = GetEmbedding(embeddim);
    target = load_target(target);

    train = (int) datadim[0] * 0.7 ; 

    printf("\n train , from 0 to %d \n" , train);
    printf("\n test ,  from %d to %d \n" , train+1 , datadim[0]-1);


    //  **************** INITIALIZE THE RNN PHASE*****************

    SimpleRNN *rnn = malloc(sizeof(SimpleRNN));
    DerivedSimpleRNN *drnn = malloc(sizeof(DerivedSimpleRNN));
    int input = 128 , hidden = 64 , output = 2;
    initialize_rnn(rnn, input, hidden, output);
    initialize_rnn_derived(rnn , drnn);

    //  ************************** TRAINING PHASE **************************
    int epoch = 20;
    float loss , acc , best_lost = 4.0  ;
    printf("\n ************************ TRAINING PHASE START ************************ \n");
    start = clock();
    for (int e = 0; e < epoch ; e++)
    {
        loss = acc = 0.0;
        printf("\nStart of epoch %d/%d \n", (e+1) , epoch);
        for (int i = 0; i < 1000; i++)
        {
            forward(rnn, data[i], datadim[1] , embedding_matrix);
            backforward(rnn, datadim[1], target[i], data[i], embedding_matrix, drnn);
            loss = loss + binary_loss_entropy(target[i], rnn->y);
            acc = accuracy(acc , target[i], rnn->y);

        }
        loss = loss/1000;
        acc = acc/1000;
        printf("--> Loss : %f  accuracy : %f \n" , loss, acc);    
        if (loss < best_lost)
        {
            best_lost = loss;  
        }
         
    }
    end = clock();
    time = (double)(end - start) / CLOCKS_PER_SEC;
    printf("\nTRAINING PHASE END IN %lf s\n" , time);
    printf("\n BEST LOST IS %lf : \n" , best_lost);

    // **************************** MODEL SAVE WITH THE BEST LOSS *****************************

    printf("\n ************************ MODEL SAVE PHASE START ************************ \n");


    // **************************** TEST MODEL PHASE *****************************

    printf("\n ************************ TEST PHASE START ************************ \n");

    float Loss = 0 ;
    int k = 0 ;
    for (int j = (train+1) ; j < datadim[0]; j++)
    {
        forward(rnn, data[j], datadim[1] , embedding_matrix);
        Loss = Loss + binary_loss_entropy(target[j], rnn->y);
        k = k + 1;
    }

    Loss = Loss / k ;

    printf("\n TEST LOST IS %lf : \n" , Loss);





    //  ************************ FREE MEMORY PHASE **********************
    deallocate_dynamic_float_matrix(embedding_matrix, embeddim[0]);
    deallocate_dynamic_int_matrix(data, datadim[0]);
    free(target);


    return 0 ;
}
