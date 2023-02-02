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
#include "algebre.h"


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


void get_data(Data *data){

	printf("\n ============= Data Summary ========== \n");
    float a;
	int b ;
    FILE *fin = fopen("../../data/data.txt" , "r");
    FILE *file = fopen("../../data/embedding.txt" , "r");
	FILE *stream = fopen("../../data/label.txt" , "r");
    if(fscanf(fin, "%d" , &data->xraw)) 
    if(fscanf(fin, "%d" , &data->xcol))
	{printf(" data shape : (%d , %d) \n" , data->xraw , data->xcol);}
	if(fscanf(file, "%d" , &data->eraw)) 
    if( fscanf(file, "%d" ,&data->ecol))
	{printf(" Embedding Matrix shape : (%d , %d) \n" , data->eraw , data->ecol);}

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
    if(fscanf(stream, "%d" , &data->xraw)) {}
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
	// printf(" Train data :  %d  \n " , data->start_val);
	// printf("Validation data from index %d to index %d  \n " , (data->start_val+1), data->end_val);
	// printf("Test  data from index %d to index %d \n " , (data->end_val+1), data->xraw);
	fclose(fin);
	fclose(file);
	fclose(stream);
}


 
