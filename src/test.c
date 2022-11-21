#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "utils.h"
#include <time.h>
#include <string.h>
#include <pthread.h>

int main()
{

    // float label[4] = {0,1,0.3,0.56};
    // double y_pred[2] = {0.60,0.40};

    // double loss = binary_loss_entropy(label, y_pred, 2);

    // printf("%lf", loss);

    // Data *data = malloc(sizeof(Data));
    // get_data(data);

    float val = 0.731; 
    float val2 = 0.729; 


    float rounded_down = rounded_float(val); 

    float rounded_down2 = rounded_float(val2); 

    if (rounded_down > rounded_down2 )
    {
        printf("BONJOUR DONALD \n");
    }
    



 
    
    return 0 ;
}
