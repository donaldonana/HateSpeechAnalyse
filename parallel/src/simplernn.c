#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <pthread.h>

#include "simplernn.h"




void training(int epoch, SimpleRNN *rnn, DerivedSimpleRNN *drnn, Data *data, int index){

	double time;
    clock_t start, end ;
    float loss , acc , best_lost = 4000.0  ;
	float *lost_list = malloc(sizeof(float)*epoch);
	float *acc_list  = malloc(sizeof(float)*epoch);
	dSimpleRNN *grnn = malloc(sizeof(dSimpleRNN));
	initialize_rnn_gradient(rnn, grnn);

    start = clock();
    for (int e = 0; e < epoch ; e++)
    {
        loss = acc = 0.0;
        printf("\nStart of epoch %d/%d \n", (e+1) , epoch);
        for (int i = 0; i < index; i++)
        {
            forward(rnn, data->X[i], data->xcol , data->embedding);
            backforward(rnn, data->xcol, data->Y[i], data->X[i], data->embedding, drnn, grnn);
            gradient_descent(rnn, grnn, 1, 0.01);
			loss = loss + binary_loss_entropy(data->Y[i], rnn->y);
            acc = accuracy(acc , data->Y[i], rnn->y);
        }
        loss = loss/index;
        acc = acc/index;
		lost_list[e] = loss;
		acc_list[e]  = acc ;
        printf("--> Loss : %f  accuracy : %f \n" , loss, acc);    
        if (rounded_float(loss) < rounded_float(best_lost))
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
 
}

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
DerivedSimpleRNN *drnn, dSimpleRNN *grnn)
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

	
	 add_matrix(grnn->d_Whx, grnn->d_Whx, drnn->dWhx, rnn->input_size, rnn->hidden_size);
	 add_matrix(grnn->d_Whh, grnn->d_Whh, drnn->dWhh, rnn->hidden_size, rnn->hidden_size);
	 add_matrix(grnn->d_Why, grnn->d_Why, drnn->dWhy, rnn->hidden_size, rnn->output_size);
	 add_vect(grnn->d_bh, grnn->d_bh, drnn->dbh, rnn->hidden_size);
	 add_vect(grnn->d_by, grnn->d_by, drnn->dby, rnn->output_size);

}

void gradient_descent(SimpleRNN *rnn, dSimpleRNN *grnn, int n, float lr){
	// Parameters Update  with SGD  o = o - lr*do
	update_matrix(rnn->W_hh, rnn->W_hh, grnn->d_Whh,  rnn->hidden_size, rnn->hidden_size, n, lr);
	update_matrix(rnn->W_hx, rnn->W_hx, grnn->d_Whx,  rnn->input_size, rnn->hidden_size, n, lr);
	update_matrix(rnn->W_yh, rnn->W_yh, grnn->d_Why,  rnn->hidden_size, rnn->output_size, n, lr);
	update_vect(rnn->b_h, rnn->b_h, grnn->d_bh, rnn->hidden_size, n, lr);
	update_vect(rnn->b_y, rnn->b_y, grnn->d_by, rnn->output_size, n, lr);
	zero_rnn_gradient(rnn, grnn);
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
void initialize_rnn_gradient(SimpleRNN *rnn, dSimpleRNN *grnn){
	grnn->d_Whx = allocate_dynamic_float_matrix(rnn->input_size, rnn->hidden_size);
	grnn->d_Whh = allocate_dynamic_float_matrix(rnn->hidden_size, rnn->hidden_size);
	grnn->d_Why = allocate_dynamic_float_matrix(rnn->hidden_size, rnn->output_size);
	grnn->d_bh = malloc(sizeof(float)*rnn->hidden_size);
	grnn->d_by = malloc(sizeof(float)*rnn->output_size);
	zero_rnn_gradient(rnn,grnn);
}

void deallocate_rnn_gradient(SimpleRNN *rnn, dSimpleRNN *grnn)
{
	deallocate_dynamic_float_matrix(grnn->d_Whx, rnn->input_size);
	deallocate_dynamic_float_matrix(grnn->d_Whh , rnn->hidden_size);
	deallocate_dynamic_float_matrix(grnn->d_Why , rnn->hidden_size);
	free(grnn->d_bh) ;
	free(grnn->d_by) ;
}

void zero_rnn_gradient(SimpleRNN *rnn, dSimpleRNN *grnn){

	initialize_vect_zero(grnn->d_bh, rnn->hidden_size);
	initialize_vect_zero(grnn->d_by, rnn->output_size);
	initialize_mat_zero(grnn->d_Whh, rnn->hidden_size, rnn->hidden_size);
	initialize_mat_zero(grnn->d_Whx, rnn->input_size, rnn->hidden_size);
	initialize_mat_zero(grnn->d_Why, rnn->hidden_size, rnn->output_size);


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

void somme_gradient( dSimpleRNN *grnn, dSimpleRNN *secondgrnn, SimpleRNN *rnn){

add_matrix(grnn->d_Whx, grnn->d_Whx, secondgrnn->d_Whx, rnn->input_size,  rnn->hidden_size);
add_matrix(grnn->d_Whh, grnn->d_Whh, secondgrnn->d_Whh, rnn->hidden_size, rnn->hidden_size);
add_matrix(grnn->d_Why, grnn->d_Why, secondgrnn->d_Why, rnn->hidden_size, rnn->output_size);
add_vect(grnn->d_bh, grnn->d_bh, secondgrnn->d_bh, rnn->hidden_size);
add_vect(grnn->d_by, grnn->d_by, secondgrnn->d_by, rnn->output_size);


}




void get_data(Data *data, int nthread){

    float a;
	int b ;
    FILE *fin = NULL;
    FILE *file = NULL;
	FILE *stream = NULL;
    fin = fopen("../python/data.txt" , "r");
    if(fscanf(fin, "%d" , &data->xraw)){printf(" xraw : %d " , data->xraw);}
    if(fscanf(fin, "%d" , &data->xcol)){printf(" xcol : %d \n" , data->xcol);}
    file = fopen("../python/embedding.txt" , "r");
	if(fscanf(file, "%d" , &data->eraw)){printf(" eraw : %d " , data->eraw);}
    if( fscanf(file, "%d" ,&data->ecol)){printf(" ecol : %d \n" , data->ecol);}

	data->embedding = allocate_dynamic_float_matrix(data->eraw, data->ecol);
	data->X = allocate_dynamic_int_matrix(data->xraw, data->xcol);
	data->Y = malloc(sizeof(int)*(data->xraw));
	// embeddind matrix
	if (file != NULL)
    {
		for (int i = 0; i < data->eraw; i++)
		{
			for (int j = 0; j < data->ecol; j++)
			{
				if(fscanf(file, "%f" , &a)){
				data->embedding[i][j] = a;
				}
			}
			
		}
    }
	// X matrix
	if (fin != NULL)
    {
		 
		for ( int i = 0; i < data->xraw; i++)
		{
			for ( int j = 0; j < data->xcol; j++)
			{
				if(fscanf(fin, "%d" , &b)){
				data->X[i][j] = b;
				}
			}

		}

    }
	// Y vector
    stream = fopen("../python/label.txt" , "r");
    if(fscanf(stream, "%d" , &data->xraw)){printf(" yraw : %d \n" , data->xraw);}
	if (stream != NULL)
    {
        int count = 0;
  		if (stream == NULL) {
    	fprintf(stderr, "Error reading file\n");
  		}
  		while (fscanf(stream, "%d", &data->Y[count]) == 1) {
      	count = count+1;
  		}
    }

	data->start_val = data->xraw * 0.7 ;
	data->end_val = data->start_val + (data->xraw * 0.1 - 1);
	printf(" Train data from index 1 to index %d  \n " , data->start_val);
	printf("Validation data from index %d to index %d  \n " , (data->start_val+1), data->end_val);
	printf("Test  data from index %d to index %d \n " , (data->end_val+1), data->xraw);

	fclose(fin);
	fclose(file);
	fclose(stream);


}

 


