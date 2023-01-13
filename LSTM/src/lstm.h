 
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

typedef struct lstm_model_parameters_t {
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
} lstm_model_parameters_t;

typedef struct lstm_model_t
{
  unsigned int X; /**< Number of input nodes */
  unsigned int N; /**< Number of neurons */
  unsigned int Y; /**< Number of output nodes */
  unsigned int S; /**< lstm_model_t.X + lstm_model_t.N */

  // Parameters
  lstm_model_parameters_t * params;

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

} lstm_model_t;

typedef struct lstm_values_cache_t {
  double* probs;
  double* probs_before_sigma;
  double* c;
  double* h;
  double* c_old;
  double* h_old;
  double* X;
  double* hf;
  double* hi;
  double* ho;
  double* hc;
  double* tanh_c_cache;
} lstm_values_cache_t;

int lstm_init_model(int X, int N, int Y, int zeros, lstm_model_parameters_t *params);

void lstm_free_model(lstm_model_t *lstm);

 

#endif
