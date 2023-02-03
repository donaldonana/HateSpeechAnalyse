#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <pthread.h>

#include "simplernn.h"


 
float testing(SimpleRNN *rnn, Data *data, int start, int end){

	float Loss = 0 ;
    int k = 0 ;
    for (int j = start ; j < end ; j++)
    {
        forward(rnn, data->X[j], data->xcol, data->embedding);
        Loss = Loss + binary_loss_entropy(data->Y[j], rnn->y);
        k = k + 1;
    }
    Loss = Loss / k ;
    // printf("\n TEST LOST IS %lf : \n" , Loss);
	return Loss ;

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

void backforward(SimpleRNN *rnn, int n, int idx, int *x, float **embedding_matrix, DerivedSimpleRNN *grad, SimpleRNN *AVGgradient)
{
	// dy = y_pred - label
    copy_vect(grad->dby, rnn->y, rnn->output_size);
    grad->dby[idx] = grad->dby[idx] - 1;

	// dWhy = last_h.T * dy 
	vect_mult(grad->dWhy , grad->dby, rnn->h[n],  rnn->hidden_size, rnn->output_size);

    // Initialize dWhh, dWhx, and dbh to zero.
	initialize_mat_zero(grad->dWhh, rnn->hidden_size, rnn->hidden_size);
	initialize_mat_zero(grad->dWhx, rnn->input_size , rnn->hidden_size);
	initialize_vect_zero(grad->dbh, rnn->hidden_size);

	// dh = np.matmul( dy , self.W_yh.T  )
	trans_mat(grad->WhyT, rnn->W_yh, rnn->hidden_size,  rnn->output_size);
	mat_mul(grad->dh , grad->dby, grad->WhyT,  rnn->output_size, rnn->hidden_size);

	for (int t = n-1; t >= 0; t--)
	{     
		// (1 - np.power( h[t+1], 2 )) * dh   
		dhraw( grad->dhraw, rnn->h[t+1] , grad->dh, rnn->hidden_size);

        // dbh += dhraw
		add_vect(grad->dbh, grad->dbh, grad->dhraw, rnn->hidden_size);

	    // dWhh += np.dot(dhraw, hs[t-1].T)
		vect_mult(grad->temp2 , grad->dhraw, rnn->h[t], rnn->hidden_size, rnn->hidden_size);
		add_matrix(grad->dWhh , grad->dWhh, grad->temp2 , rnn->hidden_size, rnn->hidden_size);

		// dWxh += np.dot(dhraw, x[t].T)
		vect_mult(grad->temp3, grad->dhraw, embedding_matrix[x[t]], rnn->input_size, rnn->hidden_size );
		add_matrix(grad->dWhx , grad->dWhx, grad->temp3, rnn->input_size, rnn->hidden_size);

		//  dh = np.matmul( dhraw, self.W_hh.T )
		trans_mat(grad->WhhT, rnn->W_hh, rnn->hidden_size,  rnn->hidden_size);
		mat_mul(grad->dh , grad->dhraw, grad->WhhT, rnn->hidden_size, rnn->hidden_size);
		
	}
	add_matrix(AVGgradient->W_hx, AVGgradient->W_hx,  grad->dWhx, rnn->input_size, rnn->hidden_size);
	add_matrix(AVGgradient->W_hh, AVGgradient->W_hh,  grad->dWhh, rnn->hidden_size, rnn->hidden_size);
	add_matrix(AVGgradient->W_yh, AVGgradient->W_yh,  grad->dWhy, rnn->hidden_size, rnn->output_size);
	add_vect(AVGgradient->b_h,    AVGgradient->b_h,   grad->dbh,  rnn->hidden_size);
	add_vect(AVGgradient->b_y,    AVGgradient->b_y,   grad->dby,  rnn->output_size);

}

void gradient_descent(SimpleRNN *rnn, SimpleRNN *AVGgradient, int n, float lr){
	// Parameters Update  with SGD  o = o - lr*do
	update_matrix(rnn->W_hh, rnn->W_hh, AVGgradient->W_hh,  rnn->hidden_size, rnn->hidden_size, n, lr);
	update_matrix(rnn->W_hx, rnn->W_hx, AVGgradient->W_hx,  rnn->input_size, rnn->hidden_size, n, lr);
	update_matrix(rnn->W_yh, rnn->W_yh, AVGgradient->W_yh,  rnn->hidden_size, rnn->output_size, n, lr);
	update_vect(rnn->b_h,    rnn->b_h,  AVGgradient->b_h, rnn->hidden_size, n, lr);
	update_vect(rnn->b_y,    rnn->b_y,  AVGgradient->b_y, rnn->output_size, n, lr);
	zero_rnn_gradient(rnn, AVGgradient);
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

void copy_rnn(SimpleRNN *rnn, SimpleRNN *secondrnn){

	secondrnn->W_hx = allocate_dynamic_float_matrix(rnn->input_size, rnn->hidden_size);
	copy_mat(secondrnn->W_hx, rnn->W_hx, rnn->input_size, rnn->hidden_size);
	secondrnn->W_hh = allocate_dynamic_float_matrix(rnn->hidden_size, rnn->hidden_size);
	copy_mat(secondrnn->W_hh, rnn->W_hh, rnn->hidden_size, rnn->hidden_size);
	secondrnn->W_yh = allocate_dynamic_float_matrix(rnn->hidden_size, rnn->output_size);
	copy_mat(secondrnn->W_yh, rnn->W_yh, rnn->hidden_size, rnn->output_size);
	secondrnn->b_h = malloc(sizeof(float)*rnn->hidden_size);
	copy_vect(secondrnn->b_h, rnn->b_h, rnn->hidden_size);
	secondrnn->b_y = malloc(sizeof(float)*rnn->output_size);
	copy_vect(secondrnn->b_y, rnn->b_y, rnn->output_size);

	secondrnn->hidden_size = rnn->hidden_size;
	secondrnn->output_size = rnn->output_size;
	secondrnn->input_size = rnn->input_size;

	secondrnn->y = malloc(sizeof(float)*rnn->output_size);
	secondrnn->h = allocate_dynamic_float_matrix(100, rnn->hidden_size);

}

void reinitialize_rnn(SimpleRNN *rnn, SimpleRNN *secondrnn){
	copy_mat(secondrnn->W_hx, rnn->W_hx, rnn->input_size, rnn->hidden_size);
	copy_mat(secondrnn->W_hh, rnn->W_hh, rnn->hidden_size, rnn->hidden_size);
	copy_mat(secondrnn->W_yh, rnn->W_yh, rnn->hidden_size, rnn->output_size);
	copy_vect(secondrnn->b_h, rnn->b_h, rnn->hidden_size);
	copy_vect(secondrnn->b_y, rnn->b_y, rnn->output_size);
}

void deallocate_rnn_derived(SimpleRNN *rnn, DerivedSimpleRNN * drnn)
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

void deallocate_rnn(SimpleRNN *rnn)
{
	deallocate_dynamic_float_matrix(rnn->W_hx, rnn->input_size);
	deallocate_dynamic_float_matrix(rnn->W_hh , rnn->hidden_size);
	deallocate_dynamic_float_matrix(rnn->W_yh , rnn->hidden_size);
	free(rnn->b_h) ;
	free(rnn->b_y) ;
	free(rnn->y) ;
    deallocate_dynamic_float_matrix(rnn->h , 100);
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
void initialize_rnn_gradient(SimpleRNN *rnn, SimpleRNN *AVGgradient){
	AVGgradient->W_hx = allocate_dynamic_float_matrix(rnn->input_size, rnn->hidden_size);
	AVGgradient->W_hh = allocate_dynamic_float_matrix(rnn->hidden_size, rnn->hidden_size);
	AVGgradient->W_yh = allocate_dynamic_float_matrix(rnn->hidden_size, rnn->output_size);
	AVGgradient->b_h = malloc(sizeof(float)*rnn->hidden_size);
	AVGgradient->b_y = malloc(sizeof(float)*rnn->output_size);
	zero_rnn_gradient(rnn, AVGgradient);
}

void deallocate_rnn_gradient(SimpleRNN *rnn, SimpleRNN *AVGgradient)
{
	deallocate_dynamic_float_matrix(AVGgradient->W_hx , rnn->input_size);
	deallocate_dynamic_float_matrix(AVGgradient->W_hh , rnn->hidden_size);
	deallocate_dynamic_float_matrix(AVGgradient->W_yh , rnn->hidden_size);
	free(AVGgradient->b_h) ;
	free(AVGgradient->b_y) ;
}

void zero_rnn_gradient(SimpleRNN *rnn, SimpleRNN *AVGgradient){

	initialize_vect_zero(AVGgradient->b_h, rnn->hidden_size);
	initialize_vect_zero(AVGgradient->b_y, rnn->output_size);

	initialize_mat_zero(AVGgradient->W_hh, rnn->hidden_size, rnn->hidden_size);
	initialize_mat_zero(AVGgradient->W_hx, rnn->input_size, rnn->hidden_size);
	initialize_mat_zero(AVGgradient->W_yh, rnn->hidden_size, rnn->output_size);


}
void initialize_rnn_derived(SimpleRNN *rnn, DerivedSimpleRNN * drnn)
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

void somme_gradient(SimpleRNN *grad, SimpleRNN *slavernn){

add_matrix(grad->W_hx, grad->W_hx, slavernn->W_hx, slavernn->input_size,  slavernn->hidden_size);
add_matrix(grad->W_hh, grad->W_hh, slavernn->W_hh, slavernn->hidden_size, slavernn->hidden_size);
add_matrix(grad->W_yh, grad->W_yh, slavernn->W_yh, slavernn->hidden_size, slavernn->output_size);
add_vect(grad->b_h,    grad->b_h, slavernn->b_h, slavernn->hidden_size);
add_vect(grad->b_y,    grad->b_y, slavernn->b_y, slavernn->output_size);

}

void modelUpdate(SimpleRNN *rnn, SimpleRNN *grad, int NUM_THREADS)
{

	update_matrix_model(rnn->W_hh, grad->W_hh, rnn->hidden_size,rnn->hidden_size, NUM_THREADS);
	update_matrix_model(rnn->W_hx, grad->W_hx, rnn->input_size, rnn->hidden_size, NUM_THREADS);
	update_matrix_model(rnn->W_yh, grad->W_yh, rnn->hidden_size,rnn->output_size, NUM_THREADS);
	update_vect_model(rnn->b_h, grad->b_h, rnn->hidden_size, NUM_THREADS);
	update_vect_model(rnn->b_y, grad->b_y, rnn->output_size, NUM_THREADS);
	zero_rnn_gradient(rnn, grad);
	
}



void print_summary(SimpleRNN* rnn, int epoch, int mini_batch, float lr, int NUM_THREADS){

	printf("\n ============= Model Summary ========== \n");
	printf("\n Model : Simple Recurrent Neural Network  \n" ) ;
	printf(" Epoch Max  :    %d    \n", epoch);
	printf(" Mini batch :    %d    \n", mini_batch);
	printf(" Learning Rate : %f    \n", lr);
	printf(" Input Size  :   %d    \n", rnn->input_size);
	printf(" Hiden Size  :   %d    \n", rnn->hidden_size);
	printf(" output Size  :  %d    \n", rnn->output_size);
	printf(" NUM THREADS  :  %d    \n", NUM_THREADS);

}



