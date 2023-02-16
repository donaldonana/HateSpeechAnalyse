
#include "gru.h"



// Inputs, Neurons, Outputs, &gru model, zeros
int gru_init_model(int X, int N, int Y, gru_rnn* gru, int zeros)
{
  int S = X + N;
  gru->X = X; /**< Number of input nodes */
  gru->N = N; /**< Number of neurons in the hiden layers */
  gru->S = S; /**< lstm_model_t.X + lstm_model_t.N */
  gru->Y = Y; /**< Number of output nodes */
  gru->probs = get_zero_vector(Y);

  if ( zeros ) {
    gru->Wz = get_zero_vector(N * S);
    gru->Wr = get_zero_vector(N * S);
    gru->Wh = get_zero_vector(N * S);
    gru->Wy = get_zero_vector(Y * N);
  } else {
    gru->Wz = get_random_vector(N * S, S);
    gru->Wr = get_random_vector(N * S, S);
    gru->Wh = get_random_vector(N * S, S);
    gru->Wy = get_random_vector(Y * N, N);
    alloc_cache_array(gru, X, N, Y, 100);
  }

  gru->bz = get_zero_vector(N);
  gru->br = get_zero_vector(N);
  gru->bh = get_zero_vector(N);
  gru->by = get_zero_vector(Y);
  gru->h_prev = get_zero_vector(N);

  gru->dldhz = get_zero_vector(N);
  gru->dldhr = get_zero_vector(S);
  gru->dldhh = get_zero_vector(N);
  gru->dldh  = get_zero_vector(N);
  gru->dldy  = get_zero_vector(Y);

  gru->dldXr = get_zero_vector(S);
  gru->dldXh = get_zero_vector(S);
  gru->dldXz = get_zero_vector(S);
 
  return 0;
}

void gru_free_model(gru_rnn* gru)
{

  free_vector(&(gru)->probs);
  free_vector(&gru->Wz);
  free_vector(&gru->Wr);
  free_vector(&gru->Wh);
  free_vector(&gru->Wy);

  free_vector(&gru->bz);
  free_vector(&gru->br);
  free_vector(&gru->bh);
  free_vector(&gru->by);
  free_vector(&gru->h_prev);

  free_vector(&gru->dldhz);
  free_vector(&gru->dldhr);
  free_vector(&gru->dldhh);
  free_vector(&gru->dldh);

  free_vector(&gru->dldXr);
  free_vector(&gru->dldXz);
  free_vector(&gru->dldXh);

  free_vector(&gru->dldy);


  free(gru);
}

// model, input, state and cache values, &probs, whether or not to apply softmax
void gru_forward(gru_rnn* model, int *x , gru_cache** cache, Data *data)
{

  double *h_prev = model->h_prev;
  int N, S, i , n, t ;
  double  *X_one_hot, *x_r_hold;
  N = model->N;
  S = model->S;
  n = (data->xcol - 1) ;

  double *tmp;
  if ( init_zero_vector(&tmp, N) ) {
    fprintf(stderr, "%s.%s.%d init_zero_vector(.., %d) failed\r\n", 
      __FILE__, __func__, __LINE__, N);
    exit(1);
  }
 
  for (t = 0; t <= n ; t++)
  {

    i = 0 ;
    X_one_hot = cache[t]->X;
    x_r_hold  = cache[t]->S;

    // concat xt whit h_hold
    while ( i < S ) 
    {
      if ( i < N ) {
        X_one_hot[i] = h_prev[i];
      } else  {
        X_one_hot[i] = data->embedding[x[t]][i-N];
      }
      ++i;
    }

    fully_connected_forward(cache[t]->hr, model->Wr, X_one_hot, model->br, N, S);
    sigmoid_forward(cache[t]->hr, cache[t]->hr, N);

    copy_vector(x_r_hold, cache[t]->hr, N);
    vectors_multiply(x_r_hold, h_prev, N);
    i = 0 ;
    while ( i < S ) 
    {
      if ( i < N ) {
        x_r_hold[i] = x_r_hold[i];
      } else  {
        x_r_hold[i] = data->embedding[x[t]][i-N];
      }
      ++i;
    }
    // ht = tanh(Wh.st + bh)
    fully_connected_forward(cache[t]->hh, model->Wh, x_r_hold, model->bh, N, S);
    tanh_forward(cache[t]->hh, cache[t]->hh, N);

    // Fully connected + sigmoid layers 
    fully_connected_forward(cache[t]->hz, model->Wz, X_one_hot, model->bz, N, S); 
    sigmoid_forward(cache[t]->hz, cache[t]->hz, N);
    
    //h =  hz * h_old + (1-hz)*hh
    copy_vector(tmp, cache[t]->hz, N);
    one_minus_vector(tmp, N);
    vectors_multiply(tmp, cache[t]->hh, N);
    copy_vector(cache[t]->h, h_prev, N);
    vectors_multiply(cache[t]->h, cache[t]->hz, N);
    vectors_add(cache[t]->h, tmp, N);

    copy_vector(cache[t]->h_old, h_prev, N);
    copy_vector(h_prev, cache[t]->h, N);

  }
  // probs = softmax ( Wy*h + by )
  fully_connected_forward(model->probs, model->Wy, cache[n]->h, model->by, model->Y, model->N);
  softmax_layers_forward(model->probs, model->probs, model->Y);
  
  free_vector(&tmp);

}

//	model, y_probabilities, y_correct, the next deltas, state and cache values, &gradients, &the next deltas
void gru_backforward(gru_rnn* model, int y_correct, int n, gru_cache** cache, gru_rnn* gradients)
{
 
  gru_cache* cache_in = NULL;
  double *dldh, *dldy, *dldhz,  *dldhh;  
  double *dldhr;
  int N, Y, S;
  
  N = model->N;
  Y = model->Y;
  S = model->S;

  double *tmp;
  if ( init_zero_vector(&tmp, N) ) {
    fprintf(stderr, "%s.%s.%d init_zero_vector(.., %d) failed\r\n", 
      __FILE__, __func__, __LINE__, N);
    exit(1);
  }

  double *bias = malloc(N*sizeof(double));
  double *weigth = malloc((N*S)*sizeof(double));

  // model cache
  dldh  = model->dldh;
  dldhz = model->dldhz;
  dldhr = model->dldhr;
  dldhh = model->dldhh;
  dldy  = model->dldy;
  copy_vector(dldy, model->probs, model->Y);

  if ( y_correct >= 0 ) {
    dldy[y_correct] -= 1.0;
  }
  
  fully_connected_backward(dldy, model->Wy, cache[n]->h , gradients->Wy, dldh, gradients->by, Y, N);

  for (int t = n ; t >= 0; t--)
  {
    cache_in = cache[t];
 
    copy_vector(dldhz, dldh, N);
    copy_vector(tmp, cache_in->h_old, N);
    vectors_substract(tmp , cache_in->hh, N);
    vectors_multiply(dldhz, tmp, N);
    sigmoid_backward(dldhz, cache_in->hz, dldhz, N);

    copy_vector(tmp, cache_in->hz, N);
    one_minus_vector(tmp, N);
    copy_vector(dldhh, dldh, N);
    vectors_multiply(dldhh, tmp, N);
    tanh_backward(dldhh, cache_in->hh, dldhh, N);

    copy_vector(tmp, dldhh, N);
    mat_mul(dldhr, tmp, model->Wh, N, S);
    tanh_backward(dldhr, cache_in->hh, dldhr, N);
    sigmoid_backward(dldhr, cache_in->hr, dldhr, N);

    fully_connected_backward(dldhz, model->Wz, cache_in->X, weigth, gradients->dldXz, bias, N, S);
    vectors_add(gradients->Wz, weigth, N*S);
    vectors_add(gradients->bz, bias, N);

    fully_connected_backward(dldhr, model->Wr, cache_in->X, weigth, gradients->dldXr, bias, N, S);
    vectors_add(gradients->Wr, weigth, N*S);
    vectors_add(gradients->br, bias, N);

    fully_connected_backward(dldhh, model->Wh, cache_in->S, weigth, gradients->dldXh, bias, N, S);
    vectors_add(gradients->Wh, weigth, N*S);
    vectors_add(gradients->bh, bias, N);

  //   // // dldXi will work as a temporary substitute for dldX (where we get extract dh_next from!)
    vectors_add(gradients->dldXz, gradients->dldXr, S);
    vectors_add(gradients->dldXz, gradients->dldXh, S);
    copy_vector(dldh, gradients->dldXz, N);

  }
   
  free_vector(&bias);
  free_vector(&weigth);
  free_vector(&tmp);

}



gru_cache*  gru_cache_container_init(int X, int N, int Y)
{
  int S = N + X;
  gru_cache* cache = e_calloc(1, sizeof(gru_cache));
  cache->h = get_zero_vector(N);
  cache->h = get_zero_vector(N);
  cache->h_old = get_zero_vector(N);
  cache->X = get_zero_vector(S);
  cache->S = get_zero_vector(S);
  cache->hz = get_zero_vector(N);
  cache->hr = get_zero_vector(N);
  cache->hh = get_zero_vector(N);
  cache->tanh_h_cache = get_zero_vector(N);

  return cache;
}


void gru_cache_container_free(gru_cache* cache_to_be_freed)
{
  free_vector(&(cache_to_be_freed)->h);
  free_vector(&(cache_to_be_freed)->h_old);
  free_vector(&(cache_to_be_freed)->h_old);
  free_vector(&(cache_to_be_freed)->X);
  free_vector(&(cache_to_be_freed)->hz);
  free_vector(&(cache_to_be_freed)->hr);
  free_vector(&(cache_to_be_freed)->hh);
  free_vector(&(cache_to_be_freed)->S);
  free_vector(&(cache_to_be_freed)->tanh_h_cache);
}


// A = A - alpha * m, m = momentum * m + ( 1 - momentum ) * dldA
void gradients_decend(gru_rnn* model, gru_rnn* gradients, float lr, int n) {

  float LR = ( 1/(float)n )*lr;

  // Computing A = A - alpha * m
  vectors_substract_scalar_multiply(model->Wy, gradients->Wy, model->Y * model->N, LR);
  vectors_substract_scalar_multiply(model->Wr, gradients->Wr, model->N * model->S, LR);
  vectors_substract_scalar_multiply(model->Wh, gradients->Wh, model->N * model->S, LR);
  vectors_substract_scalar_multiply(model->Wz, gradients->Wz, model->N * model->S, LR);

  vectors_substract_scalar_multiply(model->by, gradients->by, model->Y, LR);
  vectors_substract_scalar_multiply(model->bz, gradients->bz, model->N, LR);
  vectors_substract_scalar_multiply(model->bh, gradients->bh, model->N, LR);
  vectors_substract_scalar_multiply(model->br, gradients->br, model->N, LR);

  gru_zero_the_model(gradients);
}

 
void gru_zero_the_model(gru_rnn * model)
{
  vector_set_to_zero(model->Wy, model->Y * model->N);
  vector_set_to_zero(model->Wz, model->N * model->S);
  vector_set_to_zero(model->Wh, model->N * model->S);
  vector_set_to_zero(model->Wr, model->N * model->S);

  vector_set_to_zero(model->by, model->Y);
  vector_set_to_zero(model->br, model->N);
  vector_set_to_zero(model->bh, model->N);
  vector_set_to_zero(model->bz, model->N);
 
  vector_set_to_zero(model->dldhh, model->N);
  vector_set_to_zero(model->dldhz, model->N);
  vector_set_to_zero(model->dldhr, model->N);
  vector_set_to_zero(model->dldh, model->N);

  vector_set_to_zero(model->dldXr, model->S);
  vector_set_to_zero(model->dldXz, model->S);
  vector_set_to_zero(model->dldXh, model->S);

}


void sum_gradients(gru_rnn* gradients, gru_rnn* gradients_entry)
{
  vectors_add(gradients->Wy, gradients_entry->Wy, gradients->Y * gradients->N);
  vectors_add(gradients->Wr, gradients_entry->Wr, gradients->N * gradients->S);
  vectors_add(gradients->Wz, gradients_entry->Wz, gradients->N * gradients->S);
  vectors_add(gradients->Wh, gradients_entry->Wh, gradients->N * gradients->S);

  vectors_add(gradients->by, gradients_entry->by, gradients->Y);
  vectors_add(gradients->br, gradients_entry->br, gradients->N);
  vectors_add(gradients->bz, gradients_entry->bz, gradients->N);
  vectors_add(gradients->bh, gradients_entry->bh, gradients->N);
}

void mean_gradients(gru_rnn* gradients, double d)
{
  vectors_mean_multiply(gradients->Wy, d,  gradients->Y * gradients->N);
  vectors_mean_multiply(gradients->Wr, d,  gradients->N * gradients->S);
  vectors_mean_multiply(gradients->Wz, d,  gradients->N * gradients->S);
  vectors_mean_multiply(gradients->Wh, d,  gradients->N * gradients->S);

  vectors_mean_multiply(gradients->by, d, gradients->Y);
  vectors_mean_multiply(gradients->br, d, gradients->N);
  vectors_mean_multiply(gradients->bh, d, gradients->N);
  vectors_mean_multiply(gradients->bz, d, gradients->N);


}
 
void alloc_cache_array(gru_rnn* gru, int X, int N, int Y, int l){

  gru->cache = malloc((l)*sizeof(gru_cache));
  for (int t = 0; t < l; t++)
  {
    gru->cache[t] = gru_cache_container_init(X, N, Y);
  }

}


void copy_gru(gru_rnn* gru, gru_rnn* secondgru)
{

  // secondgru->X = gru->X;  
  // secondgru->N = gru->N;  
  // secondgru->S = gru->S;  
  // secondgru->Y = gru->Y;  
  copy_vector(secondgru->Wh, gru->Wh, gru->N*gru->S);
  copy_vector(secondgru->Wr, gru->Wr, gru->N*gru->S);
  copy_vector(secondgru->Wz, gru->Wz, gru->N*gru->S);
  copy_vector(secondgru->Wy, gru->Wy, gru->Y*gru->N);
  copy_vector(secondgru->bh, gru->bh, gru->N);
  copy_vector(secondgru->br, gru->br, gru->N);
  copy_vector(secondgru->bz, gru->bz, gru->N);
  copy_vector(secondgru->by, gru->by, gru->Y);

}


void print_summary(gru_rnn* gru, int epoch, int mini_batch, float lr, int NUM_THREADS)
{

	printf("\n ============= Model Summary ========== \n");
	printf(" Model : Gated Recurrent Unit (GRU) RNNs \n");
	printf(" Epoch Max  : %d \n", epoch);
	printf(" Mini batch : %d \n", mini_batch);
	printf(" Learning Rate : %f \n", lr);
	printf(" Input Size  : %d \n", gru->X);
	printf(" Hiden Size  : %d \n", gru->N);
	printf(" output Size  : %d \n",gru->Y);
	printf(" Num. Threads : %d \n", NUM_THREADS);

}


void update_vect_model(double *a, double *b, int l , int n)
{
	float mean = 1/(float)n ;
	for (int i = 0; i < l; i++)
	{
		a[i]= b[i]*mean ;

	}

}
 


void modelUpdate(gru_rnn *gru, gru_rnn *grad, int NUM_THREADS)
{

  update_vect_model(gru->Wz, grad->Wz, gru->S*gru->N, NUM_THREADS);
  update_vect_model(gru->Wr, grad->Wr, gru->S*gru->N, NUM_THREADS);
  update_vect_model(gru->Wh, grad->Wh, gru->S*gru->N, NUM_THREADS);
  update_vect_model(gru->Wy, grad->Wy, gru->Y*gru->N, NUM_THREADS);

  update_vect_model(gru->bh, grad->bh, gru->N, NUM_THREADS);
  update_vect_model(gru->br, grad->br, gru->N, NUM_THREADS);
  update_vect_model(gru->bh, grad->bh, gru->N, NUM_THREADS);
  update_vect_model(gru->by, grad->by, gru->Y, NUM_THREADS);

	gru_zero_the_model(grad);
}
 
void somme_gradient(gru_rnn *grad, gru_rnn *slave)
{

vectors_add(grad->Wz, slave->Wz, slave->S*slave->N);
vectors_add(grad->Wr, slave->Wr, slave->S*slave->N);
vectors_add(grad->Wh, slave->Wh, slave->S*slave->N);
vectors_add(grad->Wy, slave->Wy, slave->Y*slave->N);

vectors_add(grad->br, slave->br, slave->N);
vectors_add(grad->bh, slave->bh, slave->N);
vectors_add(grad->bz, slave->bz, slave->N);
vectors_add(grad->by, slave->by, slave->Y);

}

