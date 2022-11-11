#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "utils.h"
#include <time.h>
#include <string.h>
#include <pthread.h>

// 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

int main()
{
     srand(time(NULL));

     clock_t start, end ;
     double time;
     int epoch = 50 ;
     float **embedding_matrix = NULL;
     int **data = NULL;
     int *target = NULL;
     int embeddim[2];
     int datadim[2];
     data = GetData(datadim );
     embedding_matrix = GetEmbedding(embeddim);
     target = load_target(target);
     SimpleRNN *rnn = malloc(sizeof(SimpleRNN));
     DerivedSimpleRNN *drnn = malloc(sizeof(DerivedSimpleRNN));

     int input = 128 , hidden = 64 , output = 2;
     initialize_rnn(rnn, input, hidden, output);
     initialize_rnn_derived(rnn , drnn);
     ToEyeMatrix(rnn->W_hh, rnn->hidden_size, rnn->hidden_size);
     int i ;
     float  best_lost = 4.0 ;

     printf("\n TRAINING PHASE START \n");
     start = clock();
     for (int e = 0; e < epoch ; e++)
     {
         float loss = 0.0;
         printf("\nStart of epoch %d/%d \n", (e+1) , epoch);
         for ( i = 0; i < 4000; i++)
         {
              forward(rnn, data[i], datadim[1] , embedding_matrix);
              backforward(rnn, datadim[1], target[i], data[i], embedding_matrix, drnn);
              loss = loss + binary_loss_entropy(target[i], rnn->y);
         }
         loss = loss/4000;
         printf("  Loss : %f \n" , loss);    
         if (loss < best_lost)
         {
             best_lost = loss;  
         }
         
     }
     end = clock();
     time = (double)(end - start) / CLOCKS_PER_SEC;


     printf("\nTRAINING PHASE END IN %lf s\n" , time);
     printf("\n BEST LOST IS %lf : \n" , best_lost);


     deallocate_dynamic_float_matrix(embedding_matrix, embeddim[0]);
     deallocate_dynamic_int_matrix(data, datadim[0]);
     free(target);


     return 0 ;
}
