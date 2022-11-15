#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <pthread.h>
#define TAILLE_MAX 1000000

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include "utils.h"



/* uniform distribution, (0..1] */
float drand()   
{
  return (rand()+1.0)/(RAND_MAX+1.0);
}

/* normal distribution, centered on 0, std dev 1 */
float random_normal() 
{
  return sqrt(-2*log(drand())) * cos(2*M_PI*drand());
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


int ArgMax( float *y_pred){

	int indMax = 0;

	if (y_pred[1] > y_pred[0])
	{
		indMax = 1;
	}

	return indMax ;

}

void vector_store_as_json(float *v, int n, FILE *fo){

	if ( fo == NULL )
    return;	

	fprintf(fo, "[");

	for (int i = 0; i < n; i++)
	{
		if (i != (n-1))
		{
			fprintf(fo,"%.15f,", v[i]);	 
		}
		else{

			fprintf(fo,"%.15f,", v[i]);	 
		}
	
	}

	fprintf(fo, "]");

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

