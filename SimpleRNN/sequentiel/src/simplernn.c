
#include "simplernn.h"

// Inputs, Neurons, Outputs, &rnn model, zeros
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
    rnn->Wh = get_vector(N * S, S);
    rnn->Wy = get_vector(Y * N, N);
    alloc_cache_array(rnn, X, N, Y, 200);
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
  // Over All The Sequence
  for (t = 0; t <= n ; t++)
  {
    // Concat. h_old and xt ; [h_old;xt] 
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
    // Save hprev
    copy_vector(cache[t]->h_old, hprev, N);
    copy_vector(hprev, cache[t]->h, N);

    copy_vector(cache[t]->X, X_one_hot, S);

  }
  // probs = softmax ( Wy*h + by )
  fully_connected_forward(model->probs, model->Wy, cache[n]->h, model->by, model->Y, model->N);
  softmax_layers_forward(model->probs, model->probs, model->Y);
  
  // Free all tempory Variable
  free_vector(&hprev);
    
}


void rnn_backforward(SimpleRnn* model, double *y, int n, simple_rnn_cache** caches, SimpleRnn* gradients)
{
  simple_rnn_cache* cache = NULL;
  double *dldh, *dldy, *tmp, *weigth, *bias;
  int N, Y, S;
  N = model->N;
  Y = model->Y;
  S = model->S;
  // Tempory variable for gradient computation
  if ( 
    init_zero_vector(&tmp, N) + 
    init_zero_vector(&bias,N) +
    init_zero_vector(&weigth, N*S)  ) 
  {
    fprintf(stderr, "%s.%s.%d init_zero_vector(.., %d) failed\r\n", 
      __FILE__, __func__, __LINE__, N);
    exit(1);
  }
  // model cache
  dldh  = model->dldh;
  dldy  = model->dldy;
  copy_vector(dldy, model->probs, Y);
  vectors_substract(dldy, y, model->Y);
  // Compute dldby , dldwy and dldh
  fully_connected_backward(dldy, model->Wy, caches[n]->h , gradients->Wy, dldh, gradients->by, Y, N);
  // Compute dldwh , dldbh and update dldh
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
  // Free all tempory Variable
  free_vector(&bias);
  free_vector(&weigth);
  free_vector(&tmp);
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



float rnn_validation(SimpleRnn* rnn, Data* data)
{
  float Loss = 0.0, acc = 0.0;
  int start = data->start_val , end = data->end_val , n = 0 ;
  for (int i = start; i <= end; i++)
  {
    // Forward
    rnn_forward(rnn, data->X[i], rnn->cache, data);
    // Compute loss
    Loss = Loss + loss_entropy(data->Y[i], rnn->probs, data->ycol);
    // Compute accuracy
    acc = accuracy(acc , data->Y[i], rnn->probs, data->ycol);
    n = n + 1 ;
  }
  printf("--> Val.  Loss : %f || Val.  Accuracy : %f \n" , Loss/n, acc/n);  
  return Loss/n;

}


float rnn_test(SimpleRnn* rnn, Data* data, FILE* ft)
{
  float Loss = 0.0, acc = 0.0;
  int start = data->start_test , end = data->xraw-1, n = 0 ;
  fprintf(ft,"y,ypred\n");
  for (int i = start; i <= end; i++)
  {
    // Forward
    rnn_forward(rnn, data->X[i], rnn->cache, data);
    // Compute loss
    Loss = Loss + loss_entropy(data->Y[i], rnn->probs, data->ycol);
    ArgMax(data->Y[i], data->ycol );
    fprintf(ft,"%d,%d\n", ArgMax(data->Y[i], data->ycol) , ArgMax(rnn->probs, data->ycol ));
    // Compute accuracy
    acc = accuracy(acc , data->Y[i], rnn->probs, data->ycol);
    n = n + 1 ;
  }
  printf("\n--> Test. Loss : %f || Test. Accuracy : %f \n" , Loss/n, acc/n);  
  return Loss/n;

}


void print_summary(SimpleRnn* rnn, int epoch, int mini_batch, float lr)
{
	printf("\n ============= Model Summary ========== \n");
	printf(" Model : SIMPLE RNNs \n");
	printf(" Epoch Max  : %d \n", epoch);
	printf(" Mini batch : %d \n", mini_batch);
	printf(" Learning Rate : %f \n", lr);
	printf(" Input Size  : %d \n", rnn->X);
	printf(" Hiden Size  : %d \n", rnn->N);
	printf(" output Size  : %d \n",rnn->Y);
}


void rnn_store_net_layers_as_json(SimpleRnn* rnn, const char * filename)
{
  FILE * fp; 
  fp = fopen(filename, "w");
  if ( fp == NULL ) {
    printf("Failed to open file: %s for writing.\n", filename);
    return;
  }
    fprintf(fp, "{");
    fprintf(fp, "\n\t\"InputSize \": %d",   rnn->X);
    fprintf(fp, ",\n\t\"HiddenSize \": %d", rnn->N);
    fprintf(fp, ",\n\t\"OutputSize \": %d", rnn->Y);

    fprintf(fp, ",\n\t\"Wy\": ");
    vector_store_as_matrix_json(rnn->Wy, rnn->Y, rnn->N, fp);
    fprintf(fp, ",\n\t\"Wh\": ");
    vector_store_as_matrix_json(rnn->Wh, rnn->N, rnn->S, fp);
    
    fprintf(fp, ",\n\t\"by\": ");
    vector_store_json(rnn->by, rnn->Y, fp);
    fprintf(fp, ",\n\t\"bh\": ");
    vector_store_json(rnn->bh, rnn->N, fp);
     
    fprintf(fp, "\n}");

  fclose(fp);

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


void rnn_cache_container_free(simple_rnn_cache* cache_to_be_freed)
{
  free_vector(&(cache_to_be_freed)->h);
  free_vector(&(cache_to_be_freed)->h_old);
  free_vector(&(cache_to_be_freed)->X);
}

