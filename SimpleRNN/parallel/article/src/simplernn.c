
#include "simplernn.h"



// Inputs, Neurons, Outputs, &lstm model, zeros
int rnn_init_model(int X, int N, int Y, SimpleRnn* rnn, int zeros)
{
  int S = X + N;
  rnn->X = X; /**< Number of input nodes */
  rnn->N = N; /**< Number of neurons in the hiden layers */
  rnn->S = S; /**< lstm_model_t.X + lstm_model_t.N */
  rnn->Y = Y; /**< Number of output nodes */
  if ( zeros ) {
    rnn->Wh = get_zero_vector(N * S);
    rnn->Wy = get_zero_vector(Y * N);
  } else {
    rnn->Wh = get_random_vector(N * S, S);
    rnn->Wy = get_random_vector(Y * N, N);
    alloc_cache_array(rnn, X, N, Y, 100);
  }
  rnn->bh = get_zero_vector(N);
  rnn->by = get_zero_vector(Y);
  rnn->probs = get_zero_vector(Y);
  rnn->dldh  = get_zero_vector(N);
  rnn->dldy  = get_zero_vector(Y);
  rnn->dldXh = get_zero_vector(S);

  return 0;
}

void rnn_free_model(SimpleRnn* rnn)
{
  free_vector(&(rnn)->probs);
  free_vector(&rnn->Wh);
  free_vector(&rnn->Wy);
  free_vector(&rnn->bh);
  free_vector(&rnn->by);
  free_vector(&rnn->dldh);
  free_vector(&rnn->dldXh);
  free(rnn);
}

// model, input, state and cache values, &probs, whether or not to apply softmax
void rnn_forward(SimpleRnn* model, int *x , simple_rnn_cache** cache, Data *data)
{

  int N, S, i , n, t ;
  double  *X_one_hot;
  N = model->N;
  S = model->S;
  n = (data->xcol - 1) ;

  double *hprev;
  if ( init_zero_vector(&hprev, N) ) {
    fprintf(stderr, "%s.%s.%d init_zero_vector(.., %d) failed\r\n", 
      __FILE__, __func__, __LINE__, N);
    exit(1);
  }
 
  for (t = 0; t <= n ; t++)
  {
    i = 0 ;
    X_one_hot = cache[t]->X;
    while ( i < S ) 
    {
      if ( i < N ) {
        X_one_hot[i] = hprev[i];
      } else  {
        X_one_hot[i] = data->embedding[x[t]][i-N];
      }
      ++i;
    }
    // ht = tanh(Wh.[h_old;xt] + bh)
    fully_connected_forward(cache[t]->h, model->Wh, X_one_hot, model->bh, N, S);
    tanh_forward(cache[t]->h, cache[t]->h, N);

    copy_vector(cache[t]->h_old, hprev, N);
    copy_vector(hprev, cache[t]->h, N);

  }
  // probs = softmax ( Wy*h + by )
  fully_connected_forward(model->probs, model->Wy, cache[n]->h, model->by, model->Y, model->N);
  softmax_layers_forward(model->probs, model->probs, model->Y);
  
  free_vector(&hprev);

}


void rnn_backforward(SimpleRnn* model, int y_correct, int n, simple_rnn_cache** caches, SimpleRnn* gradients)
{
 
  simple_rnn_cache* cache = NULL;
  double *dldh, *dldy;
  int N, Y, S;
  N = model->N;
  Y = model->Y;
  S = model->S;
  
  double *bias = malloc(N*sizeof(double));
  double *weigth = malloc((N*S)*sizeof(double));

  double *tmp;
  if ( init_zero_vector(&tmp, N) ) {
    fprintf(stderr, "%s.%s.%d init_zero_vector(.., %d) failed\r\n", 
      __FILE__, __func__, __LINE__, N);
    exit(1);
  }
  // model cache
  dldh  = model->dldh;
  dldy  = model->dldy;
  copy_vector(dldy, model->probs, Y);

  if ( y_correct >= 0 ) {
    dldy[y_correct] -= 1.0;
  }

  fully_connected_backward(dldy, model->Wy, caches[n]->h , gradients->Wy, dldh, gradients->by, Y, N);
  for (int t = n ; t >= 0; t--)
  {
    cache = caches[t];
    copy_vector(tmp, dldh, N);
    tanh_backward(tmp, cache->h, tmp, N);
    fully_connected_backward(tmp, model->Wh, cache->X, weigth, gradients->dldXh, bias, N, S);
    vectors_add(gradients->Wh, weigth, N*S);
    vectors_add(gradients->bh, bias, N);
    copy_vector(dldh, gradients->dldXh, N);
  }
  free_vector(&bias);
  free_vector(&weigth);
  free_vector(&tmp);
}

void rnn_cache_container_free(simple_rnn_cache* cache_to_be_freed)
{
  free_vector(&(cache_to_be_freed)->h);
  free_vector(&(cache_to_be_freed)->h_old);
  free_vector(&(cache_to_be_freed)->X);
}

// A = A - alpha * m, m = momentum * m + ( 1 - momentum ) * dldA
void gradients_decend(SimpleRnn* model, SimpleRnn* gradients, float lr, int n) 
{
  float LR = ( 1/(float)n )*lr;
  // Computing A = A - alpha * m
  vectors_substract_scalar_multiply(model->Wy, gradients->Wy, model->Y * model->N, LR);
  vectors_substract_scalar_multiply(model->Wh, gradients->Wh, model->N * model->S, LR);
  vectors_substract_scalar_multiply(model->by, gradients->by, model->Y, LR);
  vectors_substract_scalar_multiply(model->bh, gradients->bh, model->N, LR);
}


void rnn_zero_the_model(SimpleRnn * model)
{
  vector_set_to_zero(model->Wy, model->Y * model->N);
  vector_set_to_zero(model->Wh, model->N * model->S);
  vector_set_to_zero(model->by, model->Y);
  vector_set_to_zero(model->bh, model->N);
  vector_set_to_zero(model->dldh, model->N);
  vector_set_to_zero(model->dldXh, model->S);
}


void sum_gradients(SimpleRnn* gradients, SimpleRnn* gradients_entry)
{
  vectors_add(gradients->Wy, gradients_entry->Wy, gradients->Y * gradients->N);
  vectors_add(gradients->Wh, gradients_entry->Wh, gradients->N * gradients->S);
  vectors_add(gradients->by, gradients_entry->by, gradients->Y);
  vectors_add(gradients->bh, gradients_entry->bh, gradients->N);
}

 
void rnn_validation(SimpleRnn* rnn, Data* data)
{
  printf("--->Validation \n");    

  float Loss = 0.0;
  for (int i = 1000; i < 2000; i++)
  {
    rnn_forward(rnn, data->X[i], rnn->cache, data);
    Loss = Loss + binary_loss_entropy(data->Y[i], rnn->probs);
  }
  Loss = Loss/1000;

  printf("Loss : %lf \n" , Loss);    

}


void print_summary(SimpleRnn* rnn, int epoch, int mini_batch, float lr, int NUM_THREADS)
{
	printf("\n ============= Model Summary ========== \n");
	printf(" Model : SIMPLE RNNs \n");
	printf(" Epoch Max  : %d \n", epoch);
	printf(" Mini batch : %d \n", mini_batch);
	printf(" Learning Rate : %f \n", lr);
	printf(" Num. Thread  : %d \n", NUM_THREADS);
	printf(" Input Size  : %d \n", rnn->X);
	printf(" Hiden Size  : %d \n", rnn->N);
	printf(" output Size  : %d \n",rnn->Y);
}

void alloc_cache_array(SimpleRnn* rnn, int X, int N, int Y, int l)
{
  rnn->cache = malloc((l)*sizeof(simple_rnn_cache));
  for (int t = 0; t < l; t++)
  {
    rnn->cache[t] = e_calloc(1, sizeof(simple_rnn_cache));
    rnn_cache_container_init(X, N, Y, rnn->cache[t]);
  }
}

void rnn_cache_container_init(int X, int N, int Y, simple_rnn_cache* cache )
{
  int S = N + X;
  cache->h = get_zero_vector(N);
  cache->h_old = get_zero_vector(N);
  cache->X = get_zero_vector(S);
}

void copy_rnn(SimpleRnn* rnn, SimpleRnn* secondrnn)
{

  // secondrnn->X = lstm->X;  
  // secondrnn->N = lstm->N;  
  // secondrnn->S = lstm->S;  
  // secondrnn->Y = lstm->Y;  
  copy_vector(secondrnn->Wh, rnn->Wh, rnn->N*rnn->S);
  copy_vector(secondrnn->Wy, rnn->Wy, rnn->Y*rnn->N);
  copy_vector(secondrnn->bh, rnn->bh, rnn->N);
  copy_vector(secondrnn->by, rnn->by, rnn->Y);

}

