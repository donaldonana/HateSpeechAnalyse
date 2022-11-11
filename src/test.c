#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "utils.h"
#include <time.h>
#include <string.h>
#include <pthread.h>

int main()
{

    // double label[2] = {0,1};
    // double y_pred[2] = {0.60,0.40};

    // double loss = binary_loss_entropy(label, y_pred, 2);

    // printf("%lf", loss);

    // SimpleRNN *rnn = malloc(sizeof(SimpleRNN));

    // int input = 128 , hidden = 64 , output = 2;
    //printf("///%d////", intputs);
    // initialize_rnn(rnn, input, hidden, output);
    // ToEyeMatrix(rnn->W_hh, rnn->hidden_size, rnn->hidden_size);
    // display_matrix(rnn->W_yh, hidden, output);


    int total = 41255;

    int train ;

    int test;

    train = (int) total * 0.7 ; 

    printf("\n train , from 0 to %d \n" , train);
    printf("\n test ,  from %d to %d \n" , train+1 , total);

    
    return 0 ;
}
