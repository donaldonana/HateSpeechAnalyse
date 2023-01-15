
#include "lstm.h"



// Inputs, Neurons, Outputs, &lstm model, zeros
int lstm_init_model(int X, int N, int Y, 
 lstm_model_t* lstm,  int zeros, lstm_model_parameters_t *params)
{
  int S = X + N;

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

void lstm_free_model(lstm_model_t* lstm)
{
  free_vector(&lstm->Wf);
  free_vector(&lstm->Wi);
  free_vector(&lstm->Wc);
  free_vector(&lstm->Wo);
  free_vector(&lstm->Wy);

  free_vector(&lstm->bf);
  free_vector(&lstm->bi);
  free_vector(&lstm->bc);
  free_vector(&lstm->bo);
  free_vector(&lstm->by);

  free_vector(&lstm->dldhf);
  free_vector(&lstm->dldhi);
  free_vector(&lstm->dldhc);
  free_vector(&lstm->dldho);
  free_vector(&lstm->dldc);
  free_vector(&lstm->dldh);

  free_vector(&lstm->dldXc);
  free_vector(&lstm->dldXo);
  free_vector(&lstm->dldXi);
  free_vector(&lstm->dldXf);

  free_vector(&lstm->Wfm);
  free_vector(&lstm->Wim);
  free_vector(&lstm->Wcm);
  free_vector(&lstm->Wom);
  free_vector(&lstm->Wym);

  free_vector(&lstm->bfm);
  free_vector(&lstm->bim);
  free_vector(&lstm->bcm);
  free_vector(&lstm->bom);
  free_vector(&lstm->bym);

  free(lstm);
}



// model, input, state and cache values, &probs, whether or not to apply softmax
void lstm_forward_propagate(lstm_model_t* model, double *input, 
double *h_prev, double *c_prev , lstm_values_cache_t* cache_out)
{
  int N, Y, S, i = 0;
  double *h_old, *c_old, *X_one_hot;

  h_old = h_prev ;
  c_old = c_prev ;

  N = model->N;
  Y = model->Y;
  S = model->S;

  double *tmp;
  if ( init_zero_vector(&tmp, N) ) {
    fprintf(stderr, "%s.%s.%d init_zero_vector(.., %d) failed\r\n", 
      __FILE__, __func__, __LINE__, N);
    exit(1);
  }


  X_one_hot = cache_out->X;

  while ( i < S ) {
    if ( i < N ) {
      X_one_hot[i] = h_old[i];
    } else {
      X_one_hot[i] = input[i - N];
    }
    ++i;
  }

  // Fully connected + sigmoid layers 
  fully_connected_forward(cache_out->hf, model->Wf, X_one_hot, model->bf, N, S);
  sigmoid_forward(cache_out->hf, cache_out->hf, N);

  fully_connected_forward(cache_out->hi, model->Wi, X_one_hot, model->bi, N, S);
  sigmoid_forward(cache_out->hi, cache_out->hi, N);

  fully_connected_forward(cache_out->ho, model->Wo, X_one_hot, model->bo, N, S);
  sigmoid_forward(cache_out->ho, cache_out->ho, N);

  fully_connected_forward(cache_out->hc, model->Wc, X_one_hot, model->bc, N, S);
  tanh_forward(cache_out->hc, cache_out->hc, N);

  // c = hf * c_old + hi * hc
  copy_vector(cache_out->c, cache_out->hf, N);
  vectors_multiply(cache_out->c, c_old, N);
  copy_vector(tmp, cache_out->hi, N);
  vectors_multiply(tmp, cache_out->hc, N);
  vectors_add(cache_out->c, tmp, N);

  // h = ho * tanh_c_cache
  tanh_forward(cache_out->tanh_c_cache, cache_out->c, N);
  copy_vector(cache_out->h, cache_out->ho, N);
  vectors_multiply(cache_out->h, cache_out->tanh_c_cache, N);

  // probs = softmax ( Wy*h + by )
  fully_connected_forward(cache_out->probs, model->Wy, cache_out->h, model->by, Y, N);
  softmax_layers_forward(cache_out->probs, cache_out->probs, Y, model->params->softmax_temp);


  copy_vector(cache_out->X, X_one_hot, S);


 
  free_vector(&tmp);


}


lstm_values_cache_t*  lstm_cache_container_init(int X, int N, int Y)
{
  int S = N + X;

  lstm_values_cache_t* cache = e_calloc(1, sizeof(lstm_values_cache_t));

  cache->probs = get_zero_vector(Y);
  cache->c = get_zero_vector(N);
  cache->h = get_zero_vector(N);
  cache->c_old = get_zero_vector(N);
  cache->h_old = get_zero_vector(N);
  cache->X = get_zero_vector(S);
  cache->hf = get_zero_vector(N);
  cache->hi = get_zero_vector(N);
  cache->ho = get_zero_vector(N);
  cache->hc = get_zero_vector(N);
  cache->tanh_c_cache = get_zero_vector(N);


  return cache;
}

void lstm_cache_container_free(lstm_values_cache_t* cache_to_be_freed)
{
  free_vector(&(cache_to_be_freed)->probs);
  free_vector(&(cache_to_be_freed)->c);
  free_vector(&(cache_to_be_freed)->h);
  free_vector(&(cache_to_be_freed)->c_old);
  free_vector(&(cache_to_be_freed)->h_old);
  free_vector(&(cache_to_be_freed)->X);
  free_vector(&(cache_to_be_freed)->hf);
  free_vector(&(cache_to_be_freed)->hi);
  free_vector(&(cache_to_be_freed)->ho);
  free_vector(&(cache_to_be_freed)->hc);
  free_vector(&(cache_to_be_freed)->tanh_c_cache);
}




