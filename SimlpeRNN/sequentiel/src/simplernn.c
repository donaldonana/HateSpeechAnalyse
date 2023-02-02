#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <unistd.h>
#include <pthread.h>
#include <sys/time.h>

#include "simplernn.h"

struct timeval start_t , end_t ;


void training(int epoch, SimpleRNN *rnn, gradient *drnn, SimpleRNN *AVGgradient,  Data *data, int index, int mini_batch) 
{
	printf("\n ============= Begin of Training Phase ========== \n");

	double totaltime;
	// best_lost = 4000.0  
    float loss , acc ;
	int nb_traite = 0;

    gettimeofday(&start_t, NULL);
    for (int e = 0; e < epoch ; e++)
    {
        loss = acc = 0.0;
        printf("\nStart of epoch %d/%d \n", (e+1) , epoch);

        for (int i = 0; i < index; i++)
        {
            forward(rnn, data->X[i], data->xcol , data->embedding);
            backforward(rnn, data->xcol, data->Y[i], data->X[i], data->embedding, drnn, AVGgradient);
            loss = loss + binary_loss_entropy(data->Y[i], rnn->y);
            acc = accuracy(acc , data->Y[i], rnn->y);
			if(nb_traite==mini_batch || i == (index - 1))
			{	
				gradient_descent(rnn, AVGgradient,  nb_traite, 0.01);
				nb_traite = 0;
			}
			nb_traite = nb_traite +  1 ;
        }
        loss = loss/index;
        acc  = acc/index;
        printf("--> Loss : %f  accuracy : %f \n" , loss, acc);   

        // if ((loss) < (best_lost))
        // {
        //     best_lost = loss;  
		// 	FILE *fichier = fopen("SimpleRnn.json", "w");
    	// 	save_rnn_as_json(rnn, fichier);
		// 	fclose(fichier);
    }

         
    // }
    gettimeofday(&end_t, NULL);
    totaltime = (((end_t.tv_usec - start_t.tv_usec) / 1.0e6 + end_t.tv_sec - start_t.tv_sec) * 1000) / 1000;
    printf("\nTRAINING PHASE END IN %lf s\n" , totaltime);

}

void testing(SimpleRNN *rnn, int **data, int *datadim, float **embedding_matrix, int index, int *target){

	float Loss = 0 ;
    int k = 0 ;
    for (int j = (index+1) ; j < datadim[0]; j++)
    {
        forward(rnn, data[j], datadim[1] , embedding_matrix);
        Loss = Loss + binary_loss_entropy(target[j], rnn->y);
        k = k + 1;
    }
    Loss = Loss / k ;
    printf("\n TEST LOST IS %lf : \n" , Loss);

}


void forward(SimpleRNN *rnn, int *x, int n, float **embedding_matrix){

	initialize_vect_zero(rnn->h[0], rnn->hidden_size);
    float *temp = malloc(sizeof(float)*rnn->hidden_size);
	for (int t = 0; t < n; t++)
	{
        // ht =  np.dot(xt, self.W_hx)  +  np.dot(self.h_last, self.W_hh)  + self.b_h  
		mat_mul(temp , embedding_matrix[x[t]], rnn->W_hx, rnn->input_size, rnn->hidden_size);
		mat_mul(rnn->h[t+1], rnn->h[t], rnn->W_hh, rnn->hidden_size, rnn->hidden_size);
		add_three_vect(rnn->h[t+1] , temp , rnn->b_h, rnn->h[t+1], rnn->hidden_size);
		// np.tanh(ht)
		Tanh(rnn->h[t+1], rnn->h[t+1] , rnn->hidden_size);
	}
	// y = np.dot(self.h_last, self.W_yh) + self.b_y
	mat_mul(rnn->y, rnn->h[n], rnn->W_yh,  rnn->hidden_size, rnn->output_size);
	add_vect(rnn->y, rnn->y, rnn->b_y, rnn->output_size);
	softmax(rnn->y , rnn->y , rnn->output_size);
    free(temp);
}



void backforward(SimpleRNN *rnn, int n, int idx, int *x, float **embedding_matrix, gradient *drnn, SimpleRNN *AVGgradient)
{

	// dy = y_pred - label
    copy_vect(drnn->dby, rnn->y, rnn->output_size);
    drnn->dby[idx] = drnn->dby[idx] - 1;

	// dWhy = last_h.T * dy 
	vect_mult(drnn->dWhy , drnn->dby, rnn->h[n],  rnn->hidden_size, rnn->output_size);

    // Initialize dWhh, dWhx, and dbh to zero.
	initialize_mat_zero(drnn->dWhh, rnn->hidden_size, rnn->hidden_size);
	initialize_mat_zero(drnn->dWhx, rnn->input_size , rnn->hidden_size);
	initialize_vect_zero(drnn->dbh, rnn->hidden_size);

	// dh = np.matmul( dy , self.W_yh.T  )
	trans_mat(drnn->WhyT, rnn->W_yh, rnn->hidden_size,  rnn->output_size);
	mat_mul(drnn->dh , drnn->dby, drnn->WhyT,  rnn->output_size, rnn->hidden_size);

	for (int t = n-1; t >= 0; t--)
	{     
		// (1 - np.power( h[t+1], 2 )) * dh   
		dhraw( drnn->dhraw, rnn->h[t+1] , drnn->dh, rnn->hidden_size);

        // dbh += dhraw
		add_vect(drnn->dbh, drnn->dbh, drnn->dhraw, rnn->hidden_size);

	    // dWhh += np.dot(dhraw, hs[t-1].T)
		vect_mult(drnn->temp2 , drnn->dhraw, rnn->h[t], rnn->hidden_size, rnn->hidden_size);
		add_matrix(drnn->dWhh , drnn->dWhh, drnn->temp2 , rnn->hidden_size, rnn->hidden_size);

		// dWxh += np.dot(dhraw, x[t].T)
		vect_mult(drnn->temp3, drnn->dhraw, embedding_matrix[x[t]], rnn->input_size, rnn->hidden_size );
		add_matrix(drnn->dWhx , drnn->dWhx, drnn->temp3, rnn->input_size, rnn->hidden_size);

		//  dh = np.matmul( dhraw, self.W_hh.T )
		trans_mat(drnn->WhhT, rnn->W_hh, rnn->hidden_size,  rnn->hidden_size);
		mat_mul(drnn->dh , drnn->dhraw, drnn->WhhT, rnn->hidden_size, rnn->hidden_size);
		
	}

	add_matrix(AVGgradient->W_hx, AVGgradient->W_hx, drnn->dWhx, rnn->input_size, rnn->hidden_size);
	add_matrix(AVGgradient->W_hh, AVGgradient->W_hh, drnn->dWhh, rnn->hidden_size, rnn->hidden_size);
	add_matrix(AVGgradient->W_yh, AVGgradient->W_yh, drnn->dWhy, rnn->hidden_size, rnn->output_size);
	
	add_vect(AVGgradient->b_h, AVGgradient->b_h, drnn->dbh, rnn->hidden_size);
	add_vect(AVGgradient->b_y, AVGgradient->b_y, drnn->dby, rnn->output_size);

}


void dhraw(float *dhraw, float *lasth, float *dh, int n)
{
	for (int i = 0; i < n; i++)
	{
		dhraw[i] = ( 1 - lasth[i]*lasth[i] )*dh[i];
	}
	
}


float accuracy(float acc, float y, float *y_pred) {
	int idx ;
	idx = ArgMax(y_pred);

	if (idx == y)
	{
		acc = acc + 1 ;
	}
	return acc;
}


void deallocate_rnn_derived(SimpleRNN *rnn, gradient * drnn)
{

	deallocate_dynamic_float_matrix(drnn->dWhx, rnn->input_size);
	deallocate_dynamic_float_matrix(drnn->dWhh , rnn->hidden_size);
	deallocate_dynamic_float_matrix(drnn->WhhT , rnn->hidden_size);
	deallocate_dynamic_float_matrix(drnn->dWhy , rnn->hidden_size);
	deallocate_dynamic_float_matrix(drnn->WhyT , rnn->output_size);
	free(drnn->dbh) ;
	free(drnn->dby) ;
	free(drnn->dhraw) ;
    deallocate_dynamic_float_matrix(drnn->temp2 , rnn->hidden_size);
	deallocate_dynamic_float_matrix(drnn->temp3 , rnn->hidden_size);
	free(drnn->dh) ;

}


void initialize_rnn(SimpleRNN *rnn, int input_size, int hidden_size, int output_size)
{
	rnn->input_size = input_size;
	rnn->hidden_size = hidden_size;
	rnn->output_size = output_size;
	rnn->W_hx = allocate_dynamic_float_matrix(rnn->input_size, rnn->hidden_size);
	randomly_initalialize_mat(rnn->W_hx, rnn->input_size, rnn->hidden_size);
	rnn->W_hh = allocate_dynamic_float_matrix(rnn->hidden_size, rnn->hidden_size);
    ToEyeMatrix(rnn->W_hh, rnn->hidden_size, rnn->hidden_size);
	rnn->W_yh = allocate_dynamic_float_matrix(rnn->hidden_size, rnn->output_size);
	randomly_initalialize_mat(rnn->W_yh, rnn->hidden_size, rnn->output_size);
	rnn->b_h = malloc(sizeof(float)*rnn->hidden_size);
	initialize_vect_zero(rnn->b_h, rnn->hidden_size);
	rnn->b_y = malloc(sizeof(float)*rnn->output_size);
	initialize_vect_zero(rnn->b_y, rnn->output_size);
	rnn->y = malloc(sizeof(float)*rnn->output_size);
	rnn->h = allocate_dynamic_float_matrix(100, rnn->hidden_size);

}


void initialize_rnn_derived(SimpleRNN *rnn, gradient * drnn)
{
	drnn->dWhx = allocate_dynamic_float_matrix(rnn->input_size, rnn->hidden_size);
	drnn->dWhh = allocate_dynamic_float_matrix(rnn->hidden_size, rnn->hidden_size);
	drnn->WhhT = allocate_dynamic_float_matrix(rnn->hidden_size, rnn->hidden_size);
	drnn->dWhy = allocate_dynamic_float_matrix(rnn->hidden_size, rnn->output_size);
	drnn->WhyT = allocate_dynamic_float_matrix(rnn->output_size, rnn->hidden_size);
	drnn->dbh = malloc(sizeof(float)*rnn->hidden_size);
	drnn->dby = malloc(sizeof(float)*rnn->output_size);
	drnn->dhraw = malloc(sizeof(float)*rnn->hidden_size);
    drnn->temp2 = allocate_dynamic_float_matrix(rnn->hidden_size, rnn->hidden_size);
	drnn->temp3 = allocate_dynamic_float_matrix(rnn->input_size, rnn->hidden_size);
	drnn->dh = malloc(sizeof(float)*rnn->hidden_size);
}

void initialize_rnn_gradient(SimpleRNN *AVGgradient){
	AVGgradient->W_hx = allocate_dynamic_float_matrix(AVGgradient->input_size, AVGgradient->hidden_size);
	AVGgradient->W_hh = allocate_dynamic_float_matrix(AVGgradient->hidden_size, AVGgradient->hidden_size);
	AVGgradient->W_yh = allocate_dynamic_float_matrix(AVGgradient->hidden_size, AVGgradient->output_size);
	// bias
	AVGgradient->b_h = malloc(sizeof(float)*AVGgradient->hidden_size);
	AVGgradient->b_y = malloc(sizeof(float)*AVGgradient->output_size);
	zero_rnn_gradient(AVGgradient);
}

void gradient_descent(SimpleRNN *rnn, SimpleRNN  *AVGgradient, int n, float lr){
	// Parameters Update  with SGD  o = o - lr*do
	update_matrix(rnn->W_hh, rnn->W_hh, AVGgradient->W_hh,  rnn->hidden_size, rnn->hidden_size, n, lr);
	update_matrix(rnn->W_hx, rnn->W_hx, AVGgradient->W_hx,  rnn->input_size, rnn->hidden_size, n, lr);
	update_matrix(rnn->W_yh, rnn->W_yh, AVGgradient->W_yh,  rnn->hidden_size, rnn->output_size, n, lr);
	// bias
	update_vect(rnn->b_h,  rnn->b_h,  AVGgradient->b_h,  rnn->hidden_size, n, lr);
	update_vect(rnn->b_y,  rnn->b_y,  AVGgradient->b_y,  rnn->output_size, n, lr);
	zero_rnn_gradient(AVGgradient);
}


void zero_rnn_gradient(SimpleRNN *AVGgradient)
{
	initialize_vect_zero(AVGgradient->b_h, AVGgradient->hidden_size);
	initialize_vect_zero(AVGgradient->b_y, AVGgradient->output_size);
	initialize_mat_zero(AVGgradient->W_hh, AVGgradient->hidden_size, AVGgradient->hidden_size);
	initialize_mat_zero(AVGgradient->W_hx, AVGgradient->input_size,  AVGgradient->hidden_size);
	initialize_mat_zero(AVGgradient->W_yh, AVGgradient->hidden_size, AVGgradient->output_size);
}

void save_rnn_as_json(SimpleRNN *rnn, FILE *fo){

	fprintf(fo, "{\n") ;
	fprintf(fo, "\"input_size\": %d,\n" , rnn->input_size) ;
	fprintf(fo, "\"hidden_size\": %d,\n" , rnn->hidden_size) ;
	fprintf(fo, "\"output_size\": %d,\n" , rnn->output_size) ;
	fprintf(fo, "\"Wxh\": ");
	matrix_strore_as_json(rnn->W_hx, rnn->input_size, rnn->hidden_size, fo);
	fprintf(fo, ",\n");
	fprintf(fo, "\"Whh\": ");
	matrix_strore_as_json(rnn->W_hh, rnn->hidden_size, rnn->hidden_size, fo);
	fprintf(fo, ",\n");
	fprintf(fo, "\"Wyh\": ");
	matrix_strore_as_json(rnn->W_yh, rnn->hidden_size, rnn->output_size, fo);
	fprintf(fo, ",\n");
	fprintf(fo, "\"by\": ");
	vector_store_as_json(rnn->b_y, rnn->output_size, fo);
	fprintf(fo, ",\n");
	fprintf(fo, "\"bh\": ");
	vector_store_as_json(rnn->b_h, rnn->hidden_size, fo);
	fprintf(fo, "\n");
	fprintf(fo, "}\n") ;

}



void print_summary(SimpleRNN* rnn, int epoch, int mini_batch, float lr){

	printf("\n ============= Model Summary ========== \n");
	printf(" Epoch Max  : %d \n", epoch);
	printf(" Mini batch : %d \n", mini_batch);
	printf(" Learning Rate : %f \n", 0.01);
	printf(" Input Size  : %d \n", rnn->input_size);
	printf(" Hiden Size  : %d \n", rnn->hidden_size);
	printf(" output Size  : %d \n", rnn->output_size);
}
