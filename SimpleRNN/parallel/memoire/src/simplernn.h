 
#ifndef LSTM_H
#define LSTM_H

#include <stdlib.h>

#ifdef WINDOWS

#else
#include <unistd.h>
#endif

#include <stdio.h>
#include <string.h>
#include <time.h>
#include "utilities.h"
#include "layers.h"
#include "assert.h"




typedef struct simple_rnn_cache {
  double* h_old;
  double* h;
  double* X;
} simple_rnn_cache;


typedef struct SimpleRnn
{ 
  unsigned int X; /**< Number of input nodes (input size) */
  unsigned int N; /**< Number of neurons (hiden size) */
  unsigned int Y; /**< Number of output nodes (output size) */
  unsigned int S; /**< lstm_model_t.X + lstm_model_t.N */

  // The model 
  double* Wh;
  double* Wy;
  double* bh;
  double* by;

  // rnn output probability vector 
  double* probs; 

  // cache for gradient
  double* dldh;
  double* dldXh;
  double* dldy;

  // gate and memory cell cache for time step
  simple_rnn_cache** cache;

} SimpleRnn;


int rnn_init_model(int X, int N, int Y, SimpleRnn* rnn, int zeros);

void rnn_free_model(SimpleRnn *rnn);

void rnn_forward(SimpleRnn* model, int *x ,simple_rnn_cache** cache, Data *data);

void rnn_cache_container_free(simple_rnn_cache* cache_to_be_freed);

void rnn_backforward(SimpleRnn* model , double *y, int l, simple_rnn_cache** cache_in, SimpleRnn* gradients);

void rnn_free_model(SimpleRnn* rnn);

void rnn_zero_the_model(SimpleRnn *model);

void gradients_decend(SimpleRnn* model, SimpleRnn* gradients, float lr, int n);

void rnn_training(SimpleRnn* rnn, SimpleRnn* gradient, SimpleRnn* AVGgradient,  int mini_batch_size, float lr, Data* data);

void alloc_cache_array(SimpleRnn* rnn, int X, int N, int Y, int l);

void sum_gradients(SimpleRnn* gradients, SimpleRnn* gradients_entry);

void mean_gradients(SimpleRnn* gradients, double d);

void print_summary(SimpleRnn* lstm, int epoch, int mini_batch, float lr, int NUM_THREADS);

void rnn_validation(SimpleRnn* rnn, Data* data);

void rnn_cache_container_init(int X, int N, int Y, simple_rnn_cache* cache);

void copy_rnn(SimpleRnn* rnn, SimpleRnn* secondrnn);


#endif
