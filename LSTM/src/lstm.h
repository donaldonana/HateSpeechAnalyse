 
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

#define	OPTIMIZE_ADAM                         0
#define OPTIMIZE_GRADIENT_DESCENT             1

#define LSTM_MAX_LAYERS                       10

#define BINARY_FILE_VERSION                   1

typedef struct lstm_model_parameters {
  // For progress monitoring
  double loss_moving_avg;
  // For gradient descent
  double learning_rate;
  double momentum;
  double lambda;
  double softmax_temp;
  double beta1;
  double beta2;
  int gradient_clip;
  int gradient_fit;
  int optimizer;
  int model_regularize;
  int stateful;
  int decrease_lr;
  double learning_rate_decrease;

  // General parameters
  unsigned int mini_batch_size;
  double gradient_clip_limit;
  unsigned long iterations;
  unsigned long epochs;
} lstm_model_parameters;

typedef struct lstm_rnn
{ 
  unsigned int X; /**< Number of input nodes */
  unsigned int N; /**< Number of neurons */
  unsigned int Y; /**< Number of output nodes */
  unsigned int S; /**< lstm_model_t.X + lstm_model_t.N */

  // lstm output probability vector 
  double* probs; 

  // Parameters
  lstm_model_parameters * params;

  // The model 
  double* Wf;
  double* Wi;
  double* Wc;
  double* Wo;
  double* Wy;
  double* bf;
  double* bi;
  double* bc;
  double* bo;
  double* by;

  // The model State
  // double** c;
  // double** h;

  // cache
  double* dldh;
  double* dldho;
  double* dldhf;
  double* dldhi;
  double* dldhc;
  double* dldc;

  double* dldXi;
  double* dldXo;
  double* dldXf;
  double* dldXc;

  // Gradient descent momentum
  double* Wfm;
  double* Wim;
  double* Wcm;
  double* Wom;
  double* Wym;
  double* bfm;
  double* bim;
  double* bcm;
  double* bom;
  double* bym;

} lstm_rnn;

typedef struct lstm_values_cache {
  double* c_old;
  double* h_old;
  double* c;
  double* h;
  double* X;
  double* hf;
  double* hi;
  double* ho;
  double* hc;
  double* tanh_c_cache;
} lstm_values_cache;


int lstm_init_model(int X, int N, int Y, lstm_rnn* lstm, int zeros, 
lstm_model_parameters *params);

void lstm_free_model(lstm_rnn *lstm);

void lstm_forward(lstm_rnn* model, int *x , double *h_prev, double *c_prev ,
lstm_values_cache** cache, Data *data);

lstm_values_cache*  lstm_cache_container_init(int X, int N, int Y);

void lstm_cache_container_free(lstm_values_cache* cache_to_be_freed);

void lstm_backforward(lstm_rnn* model , int y_correct, int l,
lstm_values_cache** cache_in, lstm_rnn* gradients);

float lstm_forward_bacforward(lstm_rnn* lstm, lstm_values_cache **cache_layers, Data *data);

void lstm_free_model(lstm_rnn* lstm);

void lstm_zero_the_model(lstm_rnn *model);

void gradients_decend(lstm_rnn* model, lstm_rnn* gradients);


#endif
