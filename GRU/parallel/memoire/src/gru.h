 
#ifndef gru_H
#define gru_H

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


typedef struct gru_cache {
  double* h_old;
  double* h;
  double* X;
  double* S;
  double* hz;
  double* hr;
  double* hh;
  double* tanh_h_cache;
} gru_cache;


typedef struct gru_rnn
{ 
  unsigned int X; /**< Number of input nodes */
  unsigned int N; /**< Number of neurons */
  unsigned int Y; /**< Number of output nodes */
  unsigned int S; /**< gru_model_t.X + gru_model_t.N */

  // The model 
  double* Wz;
  double* Wr;
  double* Wh;
  double* Wy;
  double* bz;
  double* br;
  double* bh;
  double* by;

  // gru output probability vector 
  double* probs; 

  // cache for gradient
  double* dldh;
  double* dldhz;
  double* dldhr;
  double* dldhh;
  double* dldXr;
  double* dldXz;
  double* dldXh;

  double* dldy;

  // gate and memory cell cache for time step
  gru_cache** cache;
  double*  h_prev;

} gru_rnn;

int gru_init_model(int X, int N, int Y, gru_rnn* gru, int zeros);

void gru_free_model(gru_rnn *gru);

void gru_forward(gru_rnn* model, int *x ,gru_cache** cache, Data *data);

void gru_cache_container_free(gru_cache* cache_to_be_freed);

void gru_backforward(gru_rnn* model , double *y, int l, gru_cache** cache_in, gru_rnn* gradients);

void gru_free_model(gru_rnn* gru);

void gru_zero_the_model(gru_rnn *model);

void gradients_decend(gru_rnn* model, gru_rnn* gradients, float lr, int n) ;

void gru_training(gru_rnn* gru, gru_rnn* gradient, gru_rnn* AVGgradient,  int mini_batch_size, float lr, Data* data);

void alloc_cache_array(gru_rnn* gru, int X, int N, int Y, int l);

void sum_gradients(gru_rnn* gradients, gru_rnn* gradients_entry);

void mean_gradients(gru_rnn* gradients, double d);

void print_summary(gru_rnn* gru, int epoch, int mini_batch, float lr, int NUM_THREADS);

void copy_gru(gru_rnn* gru, gru_rnn* secondgru);

gru_cache*  gru_cache_container_init(int X, int N, int Y);


#endif