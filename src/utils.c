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



double drand()   /* uniform distribution, (0..1] */
{
  return (rand()+1.0)/(RAND_MAX+1.0);
}

double random_normal() /* normal distribution, centered on 0, std dev 1 */
{
  return sqrt(-2*log(drand())) * cos(2*M_PI*drand());
}

void forward(SimpleRNN *rnn, int *x, int n, double **embedding_matrix){

    int idx ; 
	initialize_vect_zero(rnn->last_hs[0], rnn->hidden_size);
	// rnn->last_intput = x;
	double *temp1, *temp2, *temp3;
	temp2 = malloc(sizeof(double)*rnn->hidden_size);
	temp1 = malloc(sizeof(double)*rnn->hidden_size);
	temp3 = malloc(sizeof(double)*rnn->output_size);


	for (int i = 0; i < n; i++)
	{
        idx = x[i];
		mat_mul(temp1 , embedding_matrix[idx], rnn->W_hx, rnn->input_size, rnn->hidden_size);
		mat_mul(temp2 , rnn->last_hs[i], rnn->W_hh, rnn->hidden_size, rnn->hidden_size);
		add_vect(temp1 ,temp1, temp2, rnn->hidden_size);
		add_vect(temp1, temp1, rnn->b_h, rnn->hidden_size);
		tan_h(rnn->last_hs[i+1], rnn->hidden_size, temp1);
	}

	mat_mul(temp3 , rnn->last_hs[n], rnn->W_yh,  rnn->hidden_size, rnn->output_size);
	add_vect(temp3 , temp3, rnn->b_y, rnn->output_size);
	softmax(rnn->y , rnn->output_size, temp3);
	

	free(temp1);
	free(temp2);
	free(temp3);
	
		
}



void backforward(SimpleRNN *rnn, int n, int idx, int *x, double **embedding_matrix, 
DerivedSimpleRNN *drnn)
{

 
    copy_vect(drnn->dby, rnn->y, rnn->output_size);
    drnn->dby[idx] = drnn->dby[idx] - 1;

	vect_mult(drnn->dWhy, drnn->dby, rnn->last_hs[n],  rnn->hidden_size, rnn->output_size);

    // // Initialize dWhh,dWhx, and dbh to zero.
	initialize_mat_zero(drnn->dWhh, rnn->hidden_size, rnn->hidden_size);
	initialize_mat_zero(drnn->dWhx, rnn->input_size , rnn->hidden_size);
	initialize_vect_zero(drnn->dbh, rnn->hidden_size);

	trans_mat(drnn->WhyT, rnn->W_yh, rnn->hidden_size,  rnn->output_size);

	mat_mul(drnn->dh , drnn->dby, drnn->WhyT,  rnn->output_size, rnn->hidden_size);


    
	int idn;
	for (int t = n-1; t >= 0; t--)
	{
		vect_pow_2(drnn->temp1, rnn->last_hs[t+1], rnn->hidden_size);
		one_minus_vect(drnn->temp1, drnn->temp1, rnn->hidden_size);
		hadamar_vect(drnn->dhraw, drnn->dh, drnn->temp1, rnn->hidden_size);

        // dbh += dhraw
		add_vect(drnn->dbh, drnn->dbh, drnn->dhraw, rnn->hidden_size);

	    // dWhh += np.dot(dhraw, hs[t-1].T)
		vect_mult(drnn->temp2 , drnn->dhraw, rnn->last_hs[t], rnn->hidden_size, rnn->hidden_size);
		add_matrix(drnn->dWhh , drnn->dWhh, drnn->temp2 , rnn->hidden_size, rnn->hidden_size);

		// dWxh += np.dot(dhraw, xs[t].T)
		idn = x[t];
		vect_mult(drnn->temp3, drnn->dhraw, embedding_matrix[idn], rnn->input_size, rnn->hidden_size );
		add_matrix(drnn->dWhx , drnn->dWhx, drnn->temp3, rnn->input_size, rnn->hidden_size);

	//  dh = np.matmul( dhraw, self.W_hh.T )
		trans_mat(drnn->WhhT, rnn->W_hh, rnn->hidden_size,  rnn->hidden_size);
		mat_mul(drnn->dh , drnn->dhraw, drnn->WhhT, rnn->hidden_size, rnn->hidden_size);
		
	}





	minus_matrix(rnn->W_yh ,rnn->W_yh, drnn->dWhy , rnn->hidden_size, rnn->output_size);
	minus_matrix(rnn->W_hh ,rnn->W_hh, drnn->dWhh , rnn->hidden_size, rnn->hidden_size);
	minus_matrix(rnn->W_hx ,rnn->W_hx, drnn->dWhx , rnn->input_size, rnn->hidden_size);


	minus_vect(rnn->b_y, rnn->b_y, drnn->dby, rnn->output_size);
	minus_vect(rnn->b_h ,rnn->b_h, drnn->dbh , rnn->hidden_size);




    // free all the allocate memory
	// free(drnn->dby);
	// free(drnn->dbh);
    // free(drnn->dh);
    // free(drnn->temp1);
    // free(drnn->dhraw);
    
	// deallocate_dynamic_float_matrix(drnn->dWhy, rnn->hidden_size);
	// deallocate_dynamic_float_matrix(whyT, rnn->output_size);

	// deallocate_dynamic_float_matrix(drnn->dWhh, rnn->hidden_size); 
	// deallocate_dynamic_float_matrix(whhT, rnn->hidden_size); 

	// deallocate_dynamic_float_matrix(drnn->dWhx, rnn->input_size );      

}



void mat_mul(double *r, double* a, double** b, int n, int p) {
    // matrix a of size 1 x n (array)
    // matrix b of size n x p
    // r = a * b
    // r has size (1 x p ) --- (  (1,n)x(n,p) )
	
    int j, k;
    for (j = 0; j < p; j++) {
        r[j] = 0.0;
        for (k = 0; k < n; k++)
            r[j] += (a[k] * b[k][j]);
    }

	
}

void add_vect(double *r , double *a, double *b, int n)
{
	for (int i = 0; i < n; i++)
	{
		r[i] = a[i] + b[i] ;
	}

	
}


void copy_vect(double *a, double *b , int n)
{
	for (int i = 0; i < n; i++)
	{
		a[i] = b[i];
	}
	
}

void tan_h(double *r , int n, double* input) {
    //output[0] = 1; // Bias term


    int i;
    for (i = 0; i < n; i++) 
	{
        r[i] = tanh(input[i]); // tanh function

	}

	
}



double **GetEmbedding(FILE *fin, int row, int col) {

    double myvariable;
    int i, j ;
    double **all_vect = allocate_dynamic_float_matrix(row, col);

    for ( i = 0; i < row; i++)
    {
        for ( j = 0; j < col; j++)
        {
            fscanf(fin, "%lf" , &myvariable);
            all_vect[i][j] = myvariable;
            // printf("%.6f " , myvariable);
        }

        
    }

    fclose(fin);

    return all_vect;
     
}


int **GetData(FILE *fin, int row, int col) {

    int myvariable;
    int i, j ;
    int **all_vect = allocate_dynamic_int_matrix(row, col);

    for ( i = 0; i < row; i++)
    {
        for ( j = 0; j < col; j++)
        {
            fscanf(fin, "%d" , &myvariable);
            all_vect[i][j] = myvariable;
            // printf("%.6f " , myvariable);
        }

        
    }

    fclose(fin);

    return all_vect;
     
}

int load_target(FILE *stream, int *target )
{

	int count = 0;
  	if (stream == NULL) {
    	fprintf(stderr, "Error reading file\n");
    	return 1;
  	}
  	while (fscanf(stream, "%d", &target[count]) == 1) {
      count = count+1;
  	}

    fclose(stream);
	return 0;
    
}




double **allocate_dynamic_float_matrix(int row, int col)
{
    double **ret_val;
    int i;

    ret_val = malloc(sizeof(double *) * row);
    if (ret_val == NULL)
    {
        perror("memory allocation failure");
        exit(EXIT_FAILURE);
    }

    for (i = 0; i < row; ++i)
    {
        ret_val[i] = malloc(sizeof(double) * col);
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




void deallocate_dynamic_float_matrix(double **matrix, int row)
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

void softmax(double *r, int n, double* input) {

    //output[0] = 1; // Bias term

    int i;
    double sum = 0.0;
    for (i = 0; i < n; i++)
	{
        sum += exp(input[i]);

	}

    for (i = 0; i < n; i++) 
	{
        r[i] = exp(input[i]) / sum; // Softmax function

	}

}

double binary_loss_entropy(int idx , double *y_pred) {

    double loss;

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
	randomly_initalialize_mat(rnn->W_hh, rnn->hidden_size, rnn->hidden_size);
	
	rnn->W_yh = allocate_dynamic_float_matrix(rnn->hidden_size, rnn->output_size);
	randomly_initalialize_mat(rnn->W_yh, rnn->hidden_size, rnn->output_size);

	rnn->b_h = malloc(sizeof(double)*rnn->hidden_size);
	initialize_vect_zero(rnn->b_h, rnn->hidden_size);
	rnn->b_y = malloc(sizeof(double)*rnn->output_size);
	initialize_vect_zero(rnn->b_y, rnn->output_size);
	rnn->y = malloc(sizeof(double)*rnn->output_size);

	rnn->last_hs = allocate_dynamic_float_matrix(100, rnn->hidden_size);


}

void initialize_rnn_derived(SimpleRNN *rnn, DerivedSimpleRNN * drnn)
{

	drnn->dWhx = allocate_dynamic_float_matrix(rnn->input_size, rnn->hidden_size);

	drnn->dWhh = allocate_dynamic_float_matrix(rnn->hidden_size, rnn->hidden_size);
	drnn->WhhT = allocate_dynamic_float_matrix(rnn->hidden_size, rnn->hidden_size);

	drnn->dWhy = allocate_dynamic_float_matrix(rnn->hidden_size, rnn->output_size);
	drnn->WhyT = allocate_dynamic_float_matrix(rnn->output_size, rnn->hidden_size);

	drnn->dbh = malloc(sizeof(double)*rnn->hidden_size);
	drnn->dby = malloc(sizeof(double)*rnn->output_size);

    drnn->temp1 = malloc(sizeof(double)*rnn->hidden_size);

	drnn->dhraw = malloc(sizeof(double)*rnn->hidden_size);

    drnn->temp2 = allocate_dynamic_float_matrix(rnn->hidden_size, rnn->hidden_size);

	drnn->temp3 = allocate_dynamic_float_matrix(rnn->input_size, rnn->hidden_size);

	drnn->dh = malloc(sizeof(double)*rnn->hidden_size);

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


void randomly_initalialize_mat(double **a, int row, int col)
{

	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
		{
			a[i][j] = random_normal()/10;
		}
		
	}
	
}

void ToEyeMatrix(double **A, int row, int col) {

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



void display_matrix(double **a, int row, int col)
{
	printf("\n row = %d \t col = %d \n", row, col);
	for (int i = 0; i < row; i++)
	{
		printf("\n");
		for (int j = 0; j < col; j++)
		{
			printf(" %lf \t", a[i][j]);
		}
		
	}
	
}


void initialize_vect_zero(double *a, int n)
{
	for (int i = 0; i < n; i++)
	{
		a[i] = 0 ;
	}
	
}

void initialize_mat_zero(double **a, int row, int col)
{
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
		{
			a[i][j] = 0;
		}
		
	}
	
}

void vect_mult(double **r, double *a , double *b, int n , int m)
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


void minus_matrix(double **r, double **a , double **b, int row, int col)
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

void minus_vect(double *r, double *a, double *b, int n)
{
	
	for (int i = 0; i < n; i++)
	{
		r[i] = a[i] - (0.01)*b[i]*(0.5) ;
	}

	
}

void trans_mat(double **r, double **a, int row , int col)
{
	
	for(int i = 0; i < row; i++)
    {
        for(int j = 0; j < col; j++)
        {
            r[j][i] = a[i][j];
        }
    }

	
}

void vect_pow_2(double *r, double *a, int n)
{
	for (int i = 0; i < n; i++)
	{
		r[i] = pow(a[i], 2);
	}

	
}

void one_minus_vect(double *r, double *a , int n)
{
	for (int i = 0; i < n; i++)
	{
		r[i] = (1 - a[i]);
	}


}

void hadamar_vect(double *r, double *a, double *b, int n)
{
	for (int i = 0; i < n; i++)
	{
		r[i] = b[i] * a[i];
	}

}


void add_matrix(double **r, double **a , double **b, int row, int col)
{

	

	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
		{
			r[i][j] = a[i][j] + b[i][j];
		}
		
	}

	
}
