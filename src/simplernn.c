#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <pthread.h>

#include "simplernn.h"
#include "utils.h"




void training(int epoch, int **data, int *datadim, float **embedding_matrix, int *target,
SimpleRNN *rnn, DerivedSimpleRNN *drnn, int index){

	double time;
    clock_t start, end ;
    float loss , acc , best_lost = 4000.0  ;
	float *lost_list = malloc(sizeof(float)*epoch);
	float *acc_list  = malloc(sizeof(float)*epoch);

    start = clock();
    for (int e = 0; e < epoch ; e++)
    {
        loss = acc = 0.0;
        printf("\nStart of epoch %d/%d \n", (e+1) , epoch);
        for (int i = 0; i < index; i++)
        {
            forward(rnn, data[i], datadim[1] , embedding_matrix);
            backforward(rnn, datadim[1], target[i], data[i], embedding_matrix, drnn);
            loss = loss + binary_loss_entropy(target[i], rnn->y);
            acc = accuracy(acc , target[i], rnn->y);
        }
        loss = loss/index;
        acc = acc/index;
		lost_list[e] = loss;
		acc_list[e]  = acc ;
        printf("--> Loss : %f  accuracy : %f \n" , loss, acc);    
        if (loss < best_lost)
        {
            best_lost = loss;  
			FILE *fichier = fopen("SimpleRnn.json", "w");
    		save_rnn_as_json(rnn, fichier);
			fclose(fichier);
        }
         
    }
    end = clock();
    time = (double)(end - start) / CLOCKS_PER_SEC;
    printf("\nTRAINING PHASE END IN %lf s\n" , time);
    printf("\n BEST LOST IS %lf : \n" , best_lost);

	char axis_name[] = "loss";
	char filename[] = "DataForPlot/loss_lc_e10_b64_t4.txt";
	data_for_plot(filename, epoch, lost_list, axis_name);
	char axis_acc[] = "accuracy";
	char filename_acc[] = "DataForPlot/acc_lc_e10_b64_t4.txt";
	data_for_plot(filename_acc, epoch, acc_list, axis_acc);
	free(acc_list);
	free(lost_list);

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
    float *h1 = malloc(sizeof(float)*rnn->hidden_size);
    float *h2 = malloc(sizeof(float)*rnn->hidden_size);
	for (int t = 0; t < n; t++)
	{
        // ht =  np.dot(xt, self.W_hx)  +  np.dot(self.h_last, self.W_hh)  + self.b_h  
		mat_mul(h1 , embedding_matrix[x[t]], rnn->W_hx, rnn->input_size, rnn->hidden_size);
		mat_mul(h2, rnn->h[t], rnn->W_hh, rnn->hidden_size, rnn->hidden_size);
		add_three_vect(rnn->h[t+1] , h1 , rnn->b_h, h2, rnn->hidden_size);
		// np.tanh(ht)
		Tanh(rnn->h[t+1], rnn->h[t+1] , rnn->hidden_size);
	}
	// y = np.dot(self.h_last, self.W_yh) + self.b_y
	mat_mul(rnn->y, rnn->h[n], rnn->W_yh,  rnn->hidden_size, rnn->output_size);
	add_vect(rnn->y, rnn->y, rnn->b_y, rnn->output_size);
	softmax(rnn->y , rnn->y , rnn->output_size);
    free(h1);
    free(h2);
	
}



void backforward(SimpleRNN *rnn, int n, int idx, int *x, float **embedding_matrix, 
DerivedSimpleRNN *drnn)
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

	// Parameters Update  with SGD  o = o - lr*do
	minus_matrix(rnn->W_yh ,rnn->W_yh, drnn->dWhy , rnn->hidden_size, rnn->output_size);
	minus_matrix(rnn->W_hh ,rnn->W_hh, drnn->dWhh , rnn->hidden_size, rnn->hidden_size);
	minus_matrix(rnn->W_hx ,rnn->W_hx, drnn->dWhx , rnn->input_size, rnn->hidden_size);
	minus_vect(rnn->b_y, rnn->b_y, drnn->dby, rnn->output_size);
	minus_vect(rnn->b_h ,rnn->b_h, drnn->dbh , rnn->hidden_size);
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

