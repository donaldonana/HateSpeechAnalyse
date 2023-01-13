
#include "lstm.h"



// Inputs, Neurons, Outputs, &lstm model, zeros
int lstm_init_model(int X, int N, int Y,  int zeros, lstm_model_parameters_t *params)
{
  int S = X + N;
  lstm_model_t* lstm = e_calloc(1, sizeof(lstm_model_t));

  lstm->X = X;
  lstm->N = N;
  lstm->S = S;
  lstm->Y = Y;

  lstm->params = params;

  if ( zeros ) {
    lstm->Wf = get_zero_vector(N * S);
    lstm->Wi = get_zero_vector(N * S);
    lstm->Wc = get_zero_vector(N * S);
    lstm->Wo = get_zero_vector(N * S);
    lstm->Wy = get_zero_vector(Y * N);
  } else {
    lstm->Wf = get_random_vector(N * S, S);
    lstm->Wi = get_random_vector(N * S, S);
    lstm->Wc = get_random_vector(N * S, S);
    lstm->Wo = get_random_vector(N * S, S);
    lstm->Wy = get_random_vector(Y * N, N);
  }

  lstm->bf = get_zero_vector(N);
  lstm->bi = get_zero_vector(N);
  lstm->bc = get_zero_vector(N);
  lstm->bo = get_zero_vector(N);
  lstm->by = get_zero_vector(Y);

  lstm->dldhf = get_zero_vector(N);
  lstm->dldhi = get_zero_vector(N);
  lstm->dldhc = get_zero_vector(N);
  lstm->dldho = get_zero_vector(N);
  lstm->dldc  = get_zero_vector(N);
  lstm->dldh  = get_zero_vector(N);

  lstm->dldXc = get_zero_vector(S);
  lstm->dldXo = get_zero_vector(S);
  lstm->dldXi = get_zero_vector(S);
  lstm->dldXf = get_zero_vector(S);

  // Gradient descent momentum caches
  lstm->Wfm = get_zero_vector(N * S);
  lstm->Wim = get_zero_vector(N * S);
  lstm->Wcm = get_zero_vector(N * S);
  lstm->Wom = get_zero_vector(N * S);
  lstm->Wym = get_zero_vector(Y * N);

  lstm->bfm = get_zero_vector(N);
  lstm->bim = get_zero_vector(N);
  lstm->bcm = get_zero_vector(N);
  lstm->bom = get_zero_vector(N);
  lstm->bym = get_zero_vector(Y);

  return 0;
}

