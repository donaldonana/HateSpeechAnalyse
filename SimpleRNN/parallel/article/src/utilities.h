/*
* This file is part of the LSTM Network implementation In C made by Rickard Hallerbäck
* 
*                 Copyright (c) 2018 Rickard Hallerbäck
* 
* Permission is hereby granted, free of charge, to any person obtaining a copy of this 
* software and associated documentation files (the "Software"), 
* to deal in the Software without restriction, including without limitation the rights 
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the 
* Software, and to permit persons to whom the Software is furnished to do so, subject to 
* the following conditions:
* The above copyright notice and this permission notice shall be included in all copies 
* or substantial portions of the Software.
*
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, 
* INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR 
* PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE 
* FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR 
* OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE 
* OR OTHER DEALINGS IN THE SOFTWARE.
*/
#ifndef LSTM_UTILITIES_H
#define LSTM_UTILITIES_H

/*! \file utilities.h
    \brief Some utility functions used in the LSTM program
    
    Here are some functions that help produce the LSTM network.
*/

#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <limits.h>


typedef struct Data Data;
struct Data
{
	int xraw;
	int xcol;
	int ecol;
    int eraw;
    int yraw;
    int ycol;
	int start_val;
	int end_val;
    int start_test;
	int **X;
    double **Y;
    double **embedding;

};

void get_split_data(Data *data, float VALIDATION_SIZE);

double **allocate_dynamic_float_matrix(int row, int col);

int **allocate_dynamic_int_matrix(int row, int col);

void deallocate_dynamic_float_matrix(float **matrix, int row);

void deallocate_dynamic_int_matrix(int **matrix, int row);

// used on contigous vectors
//		A = A + B		A,		B,    l
void 	vectors_add(double*, double*, int);
void 	vectors_substract(double*, double*, int);
void 	vectors_add_scalar_multiply(double*, double*, int, double);
void 	vectors_mean_multiply(double*, double, int);
void 	vectors_substract_scalar_multiply(double*, double*, int, double);
void 	vectors_add_scalar(double*, double, int );
void 	vectors_div(double*, double*, int);
void 	vector_sqrt(double*, int);
void 	vector_store_json(double*, int, FILE *);
void 	vector_store_as_matrix_json(double*, int, int, FILE *);
//		A = A + B		A,		B,    R, C
void 	matrix_add(double**, double**, int, int);
void 	matrix_substract(double**, double**, int, int);
//		A = A*b		A,		b,    R, C
void 	matrix_scalar_multiply(double**, double, int, int);

//		A = A * B		A,		B,    l
void 	vectors_multiply(double*, double*, int);
//		A = A * b		A,		b,    l
void 	vectors_mutliply_scalar(double*, double, int);
//		A = random( (R, C) ) / sqrt(R / 2), &A, R, C
int 	init_random_matrix(double***, int, int);
//		A = 0.0s, &A, R, C
int 	init_zero_matrix(double***, int, int);
int 	free_matrix(double**, int);
//						 V to be set, Length
int 	init_zero_vector(double**, int);
int 	free_vector(double**);
//		A = B       A,		B,		length
void 	copy_vector(double*, double*, int);
double* 	get_zero_vector(int); 
double** 	get_zero_matrix(int, int);
double** 	get_random_matrix(int, int);
double* 	get_random_vector(int,int);

void  set_vector_zero(double* A, int N);
void 	matrix_set_to_zero(double**, int, int);
void 	vector_set_to_zero(double*, int);

double sample_normal(void);
double randn(double, double);

double one_norm(double*, int);

void matrix_clip(double**, double, int, int);
int vectors_fit(double*, double, int);
int vectors_clip(double*, double, int);

// I/O
void 	vector_print_min_max(char *, double *, int);
void 	vector_read(double *, int, FILE *);
void 	vector_store(double *, int, FILE *);
void 	matrix_store(double **, int, int, FILE *);  
void 	matrix_read(double **, int, int, FILE *);
void 	vector_read_ascii(double *, int, FILE *);
void 	vector_store_ascii(double *, int, FILE *);

// Memory
void*   e_calloc(size_t count, size_t size);
size_t  e_alloc_total();

/* uniform distribution, (0..1] */
float drand();

/* normal distribution, centered on 0, std dev 1 */
float random_normal() ;

float loss_entropy(double *y , double *y_pred, int n);

float binary_loss_entropy(double *y , double *y_pred, int n);

int ArgMax(double *y, int n);

float accuracy(float acc, double *y, double *y_pred, int n);

#endif

