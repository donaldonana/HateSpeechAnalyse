#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "utils.h"
#include <time.h>
#include <string.h>
#include <pthread.h>



int main()
{

     // srand(time(NULL));


     int epoch = 20 ;

     FILE *embedding = NULL;
     int row, col;
     double **embedding_matrix;
     embedding = fopen("python/embedding.txt" , "r");
     fscanf(embedding, "%d" , &row);
     printf("%d " , row);
     fscanf(embedding, "%d" , &col);
     printf("%d " , col);
     printf("\n");
     if (embedding != NULL)
     {
          embedding_matrix = GetEmbedding(embedding, row, col);

     }


     FILE *datafile = NULL;
     int drow, dcol;
     int **data;
     datafile = fopen("python/data.txt" , "r");
     fscanf(datafile, "%d" , &drow);
     printf("%d " , drow);
     fscanf(datafile, "%d" , &dcol);
     printf("%d " , dcol);
     printf("\n");
     if (datafile != NULL)
     {
          data = GetData(datafile, drow, dcol);

     }

     FILE *label = NULL;
     int lrow;
     label = fopen("python/label.txt" , "r");
     fscanf(datafile, "%d" , &lrow);
     printf("%d " , lrow);
     int *target = malloc(sizeof(int)*lrow);
     printf("\n");
     if (datafile != NULL)
     {
          load_target(label, target);

     }



    

    
    SimpleRNN *rnn = malloc(sizeof(SimpleRNN));
    DerivedSimpleRNN *drnn = malloc(sizeof(DerivedSimpleRNN));

    int input = 128 , hidden = 64 , output = 2;
    initialize_rnn(rnn, input, hidden, output);
    initialize_rnn_derived(rnn , drnn);
    ToEyeMatrix(rnn->W_hh, rnn->hidden_size, rnn->hidden_size);


    for (int e = 0; e < epoch ; e++)
    {
         double loss = 0.0;
         printf("\nStart of epoch %d/%d \n", (e+1) , epoch);
         for (int i = 0; i < 2000; i++)
         {
              forward(rnn, data[i], dcol , embedding_matrix);
              backforward(rnn, dcol, target[i], data[i], embedding_matrix, drnn);
              loss = loss + binary_loss_entropy(target[i], rnn->y);
         }
         printf("  Loss : %lf \n" , loss/2000);
         
    }





    

    



deallocate_dynamic_float_matrix(embedding_matrix, row);
deallocate_dynamic_int_matrix(data, drow);
free(target);


     return 0 ;
}
