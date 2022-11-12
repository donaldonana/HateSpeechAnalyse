#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "utils.h"
#include <time.h>
#include <string.h>
#include <pthread.h>
#define TAILLE_MAX 1000000

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif



float drand()   /* uniform distribution, (0..1] */
{
  return (rand()+1.0)/(RAND_MAX+1.0);
}

float random_normal() /* normal distribution, centered on 0, std dev 1 */
{
  return sqrt(-2*log(drand())) * cos(2*M_PI*drand());
}

void forward(SimpleRNN *rnn, int *x, int n, float **embedding_matrix){

	initialize_vect_zero(rnn->h[0], rnn->hidden_size);

	for (int t = 0; t < n; t++)
	{
        // ht =  np.dot(xt, self.W_hx)  +  np.dot(self.h_last, self.W_hh)  + self.b_h  
		mat_mul(rnn->temp1 , embedding_matrix[x[t]], rnn->W_hx, rnn->input_size, rnn->hidden_size);
		mat_mul(rnn->temp2 , rnn->h[t], rnn->W_hh, rnn->hidden_size, rnn->hidden_size);
		add_three_vect(rnn->temp1 ,rnn->temp1, rnn->b_h, rnn->temp2, rnn->hidden_size);
		// np.tanh(ht)
		Tanh(rnn->h[t+1], rnn->temp1 , rnn->hidden_size);
	}
	// y = np.dot(self.h_last, self.W_yh) + self.b_y
	mat_mul(rnn->y, rnn->h[n], rnn->W_yh,  rnn->hidden_size, rnn->output_size);
	add_vect(rnn->y, rnn->y, rnn->b_y, rnn->output_size);
	softmax(rnn->y , rnn->y , rnn->output_size);
	
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

 

void mat_mul(float *r, float* a, float** b, int n, int p) {
    // matrix a of size 1 x n (array)
    // matrix b of size n x p
    // r = a * b
    // r has size (1 x p ) --- (  (1,n)x(n,p) )
	
    int j, k;
    for (j = 0; j < p; j++) {
        r[j] = 0.0;
        for (k = 0; k < n; k++) {

            r[j] += (a[k] * b[k][j]);


		}
    }
	
}

void add_vect(float *r , float *a, float *b, int n)
{
	int i ;
	for ( i = 0; i < n; i++)
	{
		r[i] = a[i] + b[i] ;
	}
}

void add_three_vect(float *r, float *a, float *b, float *c, int n)
{

	int i ;
	for ( i = 0; i < n; i++)
	{
		r[i] = a[i] + b[i] + c[i] ;
	}
}


void copy_vect(float *a, float *b , int n)
{
	for (int i = 0; i < n; i++)
	{
		a[i] = b[i];
	}
	
}

void Tanh(float *r , float* input, int n) {
    //output[0] = 1; // Bias term

    int i;
    for (i = 0; i < n; i++) 
	{
        r[i] = tanh(input[i]); // tanh function

	}
}



float **GetEmbedding(int *dim) {

    float myvariable;
	int row, col;
    int i, j ;
    FILE *fin = NULL;
    fin = fopen("python/embedding.txt" , "r");
    if(fscanf(fin, "%d" , &row)){printf("%d " , row);}
    if( fscanf(fin, "%d" , &col)){printf("%d " , col); }
    float **embedding_matrix = allocate_dynamic_float_matrix(row, col);
    printf("\n");
    if (fin != NULL)
    {
		for ( i = 0; i < row; i++)
		{
			for ( j = 0; j < col; j++)
			{
				if(fscanf(fin, "%f" , &myvariable)){
				embedding_matrix[i][j] = myvariable;
				}
			}
			
		}
		fclose(fin);

    }

	dim[0] = row;
	dim[1] = col;


	return embedding_matrix;


}


int **GetData(int *dim) {

    int myvariable;
    int i, j ;
	int row, col;
    FILE *fin = NULL;
    fin = fopen("python/data.txt" , "r");
    if(fscanf(fin, "%d" , &row)){printf("%d " , row);}
    if(fscanf(fin, "%d" , &col)){printf("%d " , col); }
    int **data = allocate_dynamic_int_matrix(row, col);
    printf("\n");
    if (fin != NULL)
    {
		 
		for ( i = 0; i < row; i++)
		{
			for ( j = 0; j < col; j++)
			{
				if(fscanf(fin, "%d" , &myvariable)){
				data[i][j] = myvariable;
				}
			}

		}

		fclose(fin);

    }

	dim[0] = row;
	dim[1] = col;

	return data;

}

int *load_target(int *target )
{

	FILE *stream = NULL;
    int lrow;
    stream = fopen("python/label.txt" , "r");
    if(fscanf(stream, "%d" , &lrow)){printf("%d " , lrow);}
    target = malloc(sizeof(int)*lrow);
    printf("\n");
    if (stream != NULL)
    {
        int count = 0;
  		if (stream == NULL) {
    	fprintf(stderr, "Error reading file\n");
  		}
  		while (fscanf(stream, "%d", &target[count]) == 1) {
      	count = count+1;
  		}

    	fclose(stream);

    }

	return target;

	
}


float **allocate_dynamic_float_matrix(int row, int col)
{
    float **ret_val;
    int i;

    ret_val = malloc(sizeof(float *) * row);
    if (ret_val == NULL)
    {
        perror("memory allocation failure");
        exit(EXIT_FAILURE);
    }

    for (i = 0; i < row; ++i)
    {
        ret_val[i] = malloc(sizeof(float) * col);
        if (ret_val[i] == NULL)
        {
            perror("memory allocation failure");
            exit(EXIT_FAILURE);
        }
    }

    return ret_val;
}


int **allocate_dynamic_int_matrix(int row, int col)
{
    int **ret_val;
    int i;

    ret_val = malloc(sizeof(int *) * row);
    if (ret_val == NULL)
    {
        perror("memory allocation failure");
        exit(EXIT_FAILURE);
    }

    for (i = 0; i < row; ++i)
    {
        ret_val[i] = malloc(sizeof(int) * col);
        if (ret_val[i] == NULL)
        {
            perror("memory allocation failure");
            exit(EXIT_FAILURE);
        }
    }

    return ret_val;
}


void deallocate_dynamic_float_matrix(float **matrix, int row)
{
    int i;

    for (i = 0; i < row; ++i)
    {
        free(matrix[i]);
		matrix[i] = NULL;
    }
    free(matrix);
}

void deallocate_dynamic_int_matrix(int **matrix, int row)
{

    int i;

    for (i = 0; i < row; ++i)
    {
        free(matrix[i]);
		matrix[i] = NULL;
    }
    free(matrix);

}

void softmax(float *r, float* input, int n) {

    //output[0] = 1; // Bias term

    int i;
    float sum = 0.0;
    for (i = 0; i < n; i++)
	{
        sum += exp(input[i]);

	}

    for (i = 0; i < n; i++) 
	{
        r[i] = exp(input[i]) / sum; // Softmax function

	}

}

float binary_loss_entropy(int idx , float *y_pred) {

    float loss;

    loss = -1*log(y_pred[idx]);

    return loss ;
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

	rnn->temp2 = malloc(sizeof(float)*rnn->hidden_size);
	rnn->temp1 = malloc(sizeof(float)*rnn->hidden_size);


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

    drnn->temp1 = malloc(sizeof(float)*rnn->hidden_size);

	drnn->dhraw = malloc(sizeof(float)*rnn->hidden_size);

    drnn->temp2 = allocate_dynamic_float_matrix(rnn->hidden_size, rnn->hidden_size);

	drnn->temp3 = allocate_dynamic_float_matrix(rnn->input_size, rnn->hidden_size);

	drnn->dh = malloc(sizeof(float)*rnn->hidden_size);

}



void deallocate_rnn_derived(SimpleRNN *rnn, DerivedSimpleRNN * drnn)
{

	deallocate_dynamic_float_matrix(drnn->dWhx, rnn->input_size);

	deallocate_dynamic_float_matrix(drnn->dWhh , rnn->hidden_size);
	deallocate_dynamic_float_matrix(drnn->WhhT , rnn->hidden_size);

	
	deallocate_dynamic_float_matrix(drnn->dWhy , rnn->hidden_size);
	deallocate_dynamic_float_matrix(drnn->WhyT , rnn->output_size);


	free(drnn->dbh ) ;
	free(drnn->dby ) ;

    free(drnn->temp1 ) ;

	free(drnn->dhraw ) ;

    deallocate_dynamic_float_matrix(drnn->temp2 , rnn->hidden_size);

	deallocate_dynamic_float_matrix(drnn->temp3 , rnn->hidden_size);

	free(drnn->dh) ;

}


void randomly_initalialize_mat(float **a, int row, int col)
{

	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
		{
			a[i][j] = random_normal()/10;
		}
		
	}
	
}

void ToEyeMatrix(float **A, int row, int col) {

    for(int i=0;i<row;i++)                                                           
    {                                                                             
        for(int j=0;j<col;j++)                                                      
        {                                                                           
        if(i==j)                                                                  
        {                                                                         
            A[i][j] = 1;                                              
        }                                                                         
        else                                                                      
        {                                                                         
        A[i][j] = 0;                                               
        }                                                                         
        }                                                                           
    } 
}   


void display_matrix(float **a, int row, int col)
{
	printf("\n row = %d \t col = %d \n", row, col);
	for (int i = 0; i < row; i++)
	{
		printf("\n");
		for (int j = 0; j < col; j++)
		{
			printf(" %f \t", a[i][j]);
		}
		
	}
	
}


void initialize_vect_zero(float *a, int n)
{
	for (int i = 0; i < n; i++)
	{
		a[i] = 0 ;
	}
	
}

void initialize_mat_zero(float **a, int row, int col)
{
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
		{
			a[i][j] = 0;
		}
		
	}
	
}

void vect_mult(float **r, float *a , float *b, int n , int m)
{
	// matrix a of size 1 x m 
    // matrix b of size n x 1
    // result = b * a
    // matrix result of size n x m (array)
    

	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < m; j++)
		{
			r[i][j] = a[j]*b[i]; 
		}
		
	}
	
}


void minus_matrix(float **r, float **a , float **b, int row, int col)
{


	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
		{
			r[i][j] = a[i][j] - (0.01)*b[i][j]*(0.5);
			//r[i][j] = 0.0; 
		}
	}

}

void minus_vect(float *r, float *a, float *b, int n)
{
	
	for (int i = 0; i < n; i++)
	{
		r[i] = a[i] - (0.01)*b[i]*(0.5) ;
	}

	
}

void trans_mat(float **r, float **a, int row , int col)
{
	
	for(int i = 0; i < row; i++)
    {
        for(int j = 0; j < col; j++)
        {
            r[j][i] = a[i][j];
        }
    }

	
}

void dhraw(float *dhraw, float *lasth, float *dh, int n)
{
	for (int i = 0; i < n; i++)
	{
		dhraw[i] = ( 1 - lasth[i]*lasth[i] )*dh[i];
	}

	
}

void one_minus_vect(float *r, float *a , int n)
{
	for (int i = 0; i < n; i++)
	{
		r[i] = (1 - a[i]);
	}


}

void hadamar_vect(float *r, float *a, float *b, int n)
{
	for (int i = 0; i < n; i++)
	{
		r[i] = b[i] * a[i];
	}

}


void add_matrix(float **r, float **a , float **b, int row, int col)
{

	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
		{
			r[i][j] = a[i][j] + b[i][j];
		}
		
	}

	
}

float test( SimpleRNN *rnn , int **x, int *y, float **embedding_matrix, 
int start, int end, int n ) { 

	float loss = 0;
	int k = 0;

	for (int i = start; i < end; i++)
	{
		forward(rnn, x[i], n, embedding_matrix);
        loss = loss + binary_loss_entropy(y[i], rnn->y);
		k = k + 1;

	}

	loss = loss / k ;

	return loss ;
	


}
