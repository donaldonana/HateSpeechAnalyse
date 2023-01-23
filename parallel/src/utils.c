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

float rounded_float(float val){

	return (floorf(val * 100) / 100 );

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

void copy_mat(float **a, float **b, int row ,int col){

	for (int  i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
		{
			a[i][j] = b[i][j];
		}
		
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


void update_matrix(float **r, float **a , float **b, int row, int col, int n, float lr)
{
	float mean = 1/(float)n ;
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
		{
			r[i][j] = a[i][j] - (lr)*b[i][j]*(0.5)*mean;
		}
	}
}

void update_vect(float *r, float *a, float *b, int col , int n, float lr)
{
	float mean = 1/(float)n ;
	for (int i = 0; i < col; i++)
	{
		r[i] = a[i] - (lr)*b[i]*(0.5)*mean ;
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
			fprintf(fo,"%.15f", v[i]);	 
		}
	
	}
	fprintf(fo, "]");
}

void matrix_strore_as_json(float **m, int row, int col, FILE *fo){

	fprintf(fo, "[");

	for (int i = 0; i < row; i++)
	{
		fprintf(fo, "[");

		for (int j = 0; j < col; j++)
		{
			if (j != (col - 1))
			{
				fprintf(fo,"%.15f,", m[i][j]);	 
			}
			else{
			fprintf(fo,"%.15f", m[i][j]);	 
			}
			
		}
		
		if (i != (row - 1))
		{
			fprintf(fo, "],");
 
		}
		else{
			fprintf(fo, "]");
	 
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


void data_for_plot(char *filename, int epoch, float *axis, char *axis_name){

 	FILE* fichier = NULL;
	fichier = fopen(filename, "w");
	if (fichier != NULL)
	{
 		// printf("%s", filename);
		fprintf(fichier,"epoch,%s\n", axis_name);

		for (int k = 0; k < epoch; k++)
		{
			fprintf(fichier,"%d,%f\n", (k+1), axis[k]);

		}
	}
	else
	{
		// On affiche un message d'erreur si on veut
		printf("Impossible d'ouvrir le fichier test.txt");
	}
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

 

 