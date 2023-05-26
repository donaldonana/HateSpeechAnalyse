 
/* 
* Dealing with common vector operations 
*
* ==================== WARNING ====================
*   The caller should have thought about the memory 
*    allocation, these functions assumes that 
*    everything is OK. If not used with care, 
*   prohibted reads/writes might occur.
* =================================================
*
*/
#include "utilities.h"



// used on contigous vectors
void  vectors_add(double* A, double* B, int L)
{
  int l = 0;
  while ( l < L ) {
    A[l] += B[l];
    ++l;
  }
}

void  vectors_add_scalar(double* A, double B, int L)
{
  int l = 0;
  while ( l < L ) {
    A[l] += B;
    ++l;
  }
}

void  vectors_mean_multiply(double* A, double d, int L)
{
  int l = 0;
  while ( l < L ) {
    A[l] *= d;
    ++l;
  }
}

// A = A + (B * s)
void  vectors_add_scalar_multiply(double* A, double* B, int L, double s)
{
  int l = 0;
  while ( l < L ) {
    A[l] += B[l] * s;
    ++l;
  }
}

void  vectors_substract(double* A, double* B, int L)
{
  int l = 0;
  while ( l < L ) {
    A[l] -= B[l];
    ++l;
  }
}

void  vectors_div(double* A, double* B, int L)
{
  int l = 0;
  while ( l < L ) {
    A[l] /= B[l];
    ++l;
  }
}

void  vector_sqrt(double* A, int L)
{
  int l = 0;
  while ( l < L ) {
    A[l] = sqrt(A[l]);
    ++l;
  }
}
// A = A - (B * s)
void  vectors_substract_scalar_multiply(double* A, double* B, int L, double s)
{
  int l = 0;
  while ( l < L ) {
    A[l] -= B[l]*s;
    ++l;
  }
}


void  vectors_multiply(double* A, double* B, int L)
{
  int l = 0;
  while ( l < L ) {
    A[l] *= B[l];
    ++l;
  }
}
void  vectors_mutliply_scalar(double* A, double b, int L)
{
  int l = 0;
  while ( l < L ) {
    A[l] *= b;
    ++l;
  }
} 

int   init_random_matrix(double*** A, int R, int C)
{
  int r = 0, c = 0;

  *A = e_calloc(R, sizeof(double*));

  while ( r < R ) {
    (*A)[r] = e_calloc(C, sizeof(double));
    ++r;
  }

  r = 0, c = 0;

  while ( r < R ) {
    c = 0;
    while ( c < C ) {
      (*A)[r][c] =  randn(0,1) / sqrt( R ); 
      ++c;
    }
    ++r;
  }

  return 0;
}

double*   get_random_vector(int L, int R) {
  
  int l = 0;
  double *p;
  p = e_calloc(L, sizeof(double));

  while ( l < L ) {
    p[l] = random_normal()/10;
    ++l;
  }

  return p;

}

double**  get_random_matrix(int R, int C)
{
  int r = 0, c = 0;
  double ** p;
  p = e_calloc(R, sizeof(double*));

  while ( r < R ) {
    p[r] = e_calloc(C, sizeof(double));
    ++r;
  }

  r = 0, c = 0;

  while ( r < R ) {
    c = 0;
    while ( c < C ) {
      p[r][c] =  ((( (double) rand() ) / RAND_MAX) ) / sqrt( R / 2.0 ); 
      ++c;
    }
    ++r;
  }

  return p;
}

double**  get_zero_matrix(int R, int C)
{
  int r = 0, c = 0;
  double ** p;
  p = e_calloc(R, sizeof(double*));

  while ( r < R ) {
    p[r] = e_calloc(C, sizeof(double));

    ++r;
  }

  r = 0, c = 0;

  while ( r < R ) {
    c = 0;
    while ( c < C ) {
      p[r][c] =  0.0;
      ++c;
    }
    ++r;
  }

  return p;
}

int   init_zero_matrix(double*** A, int R, int C)
{
  int r = 0, c = 0;

  *A = e_calloc(R, sizeof(double*));

  while ( r < R ) {
    (*A)[r] = e_calloc(C, sizeof(double));

    ++r;
  }

  r = 0, c = 0;

  while ( r < R ) {
    c = 0;
    while ( c < C ) {
      (*A)[r][c] = 0.0;
      ++c;
    }
    ++r;
  }

  return 0;
}

int   free_matrix(double** A, int R)
{
  int r = 0;
  while ( r < R ) {
    free(A[r]);
    ++r;  
  }
  free(A);
  return 0;
}

int   init_zero_vector(double** V, int L) 
{
  int l = 0;
  *V = e_calloc(L, sizeof(double));

  while ( l < L ) {
    (*V)[l] = 0.0;
    ++l;
  }
 
  return 0;
}

double*   get_zero_vector(int L) 
{
  int l = 0;
  double *p;
  p = e_calloc(L, sizeof(double));

  while ( l < L ) {
    p[l] = 0.0;
    ++l;
  }

  return p;
}

int   free_vector(double** V)
{
  free(*V);
  *V = NULL;
  return 0;
}

void  copy_vector(double* A, double* B, int L)
{
  int l = 0;

  while ( l < L ) {
    A[l] = B[l];
    ++l;
  }
}

void  set_vector_zero(double* A, int N)
{
  for (int i = 0; i < N; i++)
  {
    A[i] = 0.0 ;
  }
  
}

void  matrix_add(double** A, double** B, int R, int C)
{
  int r = 0, c = 0;

  while ( r < R ) {
    c = 0;
    while ( c < C ) {
      A[r][c] += B[r][c];
      ++c;
    }
    ++r;
  }
}

void  vector_set_to_zero(double* V, int L )
{
  int l = 0;
  while ( l < L )
    V[l++] = 0.0;
}


void  matrix_set_to_zero(double** A, int R, int C)
{
  int r = 0, c = 0;

  while ( r < R ) {
    c = 0;
    while ( c < C ) {
      A[r][c] = 0.0;
      ++c;
    }
    ++r;
  }
}

void  matrix_substract(double** A, double** B, int R, int C)
{
  int r = 0, c = 0;

  while ( r < R ) {
    c = 0;
    while ( c < C ) {
      A[r][c] -= B[r][c];
      ++c;
    }
    ++r;
  }
}

void  matrix_scalar_multiply(double** A, double b, int R, int C)
{
  int r = 0, c = 0;

  while ( r < R ) {
    c = 0;
    while ( c < C ) {
      A[r][c] *= b;
      ++c;
    }
    ++r;
  }
}
void  matrix_clip(double** A, double limit, int R, int C)
{
  int r = 0, c = 0;

  while ( r < R ) {
    c = 0;
    while ( c < C ) {
      if ( A[r][c] > limit )
        A[r][c] = limit;
      else if ( A[r][c] < -limit )
        A[r][C] = -limit;
      ++c;
    }
    ++r;
  }
}

double one_norm(double* V, int L)
{
  int l = 0;
  double norm = 0.0;
  while ( l < L ) {
    norm += fabs(V[l]);
    ++l;
  }

  return norm;
}

int   vectors_fit(double* V, double limit, int L)
{
  int l = 0;
  int msg = 0;
  double norm;
  while ( l < L ) {
    if ( V[l] > limit || V[l] < -limit ) {
      msg = 1;
      norm = one_norm(V, L);
      break;
    }
    ++l;
  }

  if ( msg )
    vectors_mutliply_scalar(V, limit / norm, L);

  return msg;
}

int   vectors_clip(double* V, double limit, int L)
{
  int l = 0;
  int msg = 0;
  while ( l < L ) {
    if ( V[l] > limit ) {
      msg = 1;
      V[l] = limit;
    } else if ( V[l] < -limit ) {
      msg = 1;
      V[l] = -limit;
    }
    ++l;
  }

  return msg;
}

void  matrix_store(double ** A, int R, int C, FILE * fp) 
{
  int r = 0, c = 0;
  size_t i = 0;
  char *p;

  while ( r < R ) {
    c = 0;
    while ( c < C ) {
      i = 0; p = (char*)&A[r][c];
      while ( i < sizeof(double) ) {
        fputc(*(p), fp);
        ++i; ++p;
      }
      ++c;
    }
    ++r;
  }

}

void  vector_print_min_max(char *name, double *V, int L)
{
  int l = 0;
  double min = 100;
  double max = -100;
  while ( l < L ) {
    if ( V[l] > max )
      max = V[l];
    if ( V[l] < min )
      min = V[l];
    ++l;
  }
  printf("%s min: %.10lf, max: %.10lf\n", name, min, max);
}

void  matrix_read(double ** A, int R, int C, FILE * fp) 
{
  int r = 0, c = 0;
  size_t i = 0;
  char *p;
  double value;

  while ( r < R ) {
    c = 0;
    while ( c < C ) {
      i = 0; p = (char*)&value;
      while ( i < sizeof(double) ) {
        *(p) = fgetc(fp);
        ++i; ++p;
      }
      A[r][c] = value;
      ++c;
    }
    ++r;
  }

}

void  vector_store(double* V, int L, FILE * fp)
{
  int l = 0;
  size_t i = 0;
  char *p;

  while ( l < L ) {
    i = 0; p = (char*)&V[l];
    while ( i < sizeof(double) ) {
      fputc(*(p), fp);
      ++i; ++p;
    }
    ++l;
  }
}

void  vector_read(double * V, int L, FILE * fp) 
{
  int l = 0;
  size_t i = 0;
  char *p;
  double value;

  while ( l < L ) {
    i = 0; p = (char*)&value;
    while ( i < sizeof(double) ) {
      *(p) = fgetc(fp);
      ++i; ++p;
    }
    V[l] = value;
    ++l;
  }

}

void  vector_store_ascii(double* V, int L, FILE * fp)
{
  int l = 0;

  while ( l < L ) {
    fprintf(fp, "%.20lf\r\n", V[l]);
    ++l;
  }
}

void  vector_read_ascii(double * V, int L, FILE * fp)
{
  int l = 0;

  while ( l < L ) {
    if ( fscanf(fp, "%lf", &V[l]) <= 0 ) {
      fprintf(stderr, "%s.%s Failed to read file\r\n",
        __FILE__, __func__);
      exit(1);
    }
    ++l;
  }

}

/*
*   This function is used to store a JSON file representation
*   of a LSTM neural network that can be read by an HTML application.
*/
void  vector_store_as_matrix_json(double* V, int R, int C, FILE * fp)
{
  int r = 0, c = 0;

  if ( fp == NULL )
    return; // No file, nothing to do. 

  fprintf(fp, "[");

  r = 0;

  while ( r < R ) {

    if ( r > 0 )
      fprintf(fp, ",");

    fprintf(fp,"[");

    c = 0;
    while ( c < C ) {

      if ( c > 0 )
        fprintf(fp, ",");

      fprintf(fp,"%.15f", V[r*C + c]);

      ++c;
    }

    fprintf(fp,"]");

    ++r;
  }

  fprintf(fp, "]");
}


/*
*   This function is used to store a JSON file representation
*   of a LSTM neural network that can be read by an HTML application.
*/
void  vector_store_json(double* V, int L, FILE * fp)
{
  int l = 0;

  if ( fp == NULL )
    return; // No file, nothing to do. 

  fprintf(fp, "[");

  while ( l < L ) {

    if ( l > 0 )
      fprintf(fp, ",");

    fprintf(fp,"%.15f", V[l]);

    ++l;
  }

  fprintf(fp, "]");
}

/*
* Gaussian generator: https://phoxis.org/2013/05/04/generating-random-numbers-from-normal-distribution-in-c/
*/
double
randn (double mu, double sigma)
{
  double U1, U2, W, mult;
  static double X1, X2;
  static int call = 0;
 
  if (call == 1)
  {
    call = !call;
    return (mu + sigma * (double) X2);
  }

  do {
    U1 = -1 + ((double) rand () / RAND_MAX) * 2;
    U2 = -1 + ((double) rand () / RAND_MAX) * 2;
    W = pow (U1, 2) + pow (U2, 2);
  } while ( W >= 1 || W == 0 );
 
  mult = sqrt ((-2 * log (W)) / W);
  X1 = U1 * mult;
  X2 = U2 * mult;
 
  call = !call;
 
  return (mu + sigma * (double) X1);
}

double sample_normal() {
  double u = ((double) rand() / (RAND_MAX)) * 2 - 1;
  double v = ((double) rand() / (RAND_MAX)) * 2 - 1;
  double r = u * u + v * v;
  if (r == 0 || r > 1)
    return sample_normal();
  double c = sqrt(-2 * log(r) / r);
  return u * c;
}

/* Memory related utilities */
static size_t alloc_mem_tot = 0;
void*   e_calloc(size_t count, size_t size)
{
  void *p = calloc(count, size);
  if ( p == NULL ) {
    /* Failed to allocate this memory will exit */
    fprintf(stderr, "%s error: Failed to allocate %zu bytes, having allocated %zu in total already.\n", 
      __func__, count*size, alloc_mem_tot);
    exit(1);
  }
  alloc_mem_tot += count*size;
  return p;
}

size_t  e_alloc_total()
{
  return alloc_mem_tot;
}




double **allocate_dynamic_float_matrix(int row, int col)
{
    double **ret_val;
    int i;

    ret_val = malloc(sizeof(double *) * row);
    if (ret_val == NULL)
    {
        perror("memory allocation failure");
        exit(EXIT_FAILURE);
    }

    for (i = 0; i < row; ++i)
    {
        ret_val[i] = malloc(sizeof(double) * col);
        if (ret_val[i] == NULL)
        {
            perror("memory allocation failure");
            exit(EXIT_FAILURE);
        }
    }

    return ret_val;
}


int **allocate_dynamic_int_matrix(int row, int col)
{
    int **ret_val;
    int i;

    ret_val = malloc(sizeof(int *) * row);
    if (ret_val == NULL)
    {
        perror("memory allocation failure");
        exit(EXIT_FAILURE);
    }

    for (i = 0; i < row; ++i)
    {
        ret_val[i] = malloc(sizeof(int) * col);
        if (ret_val[i] == NULL)
        {
            perror("memory allocation failure");
            exit(EXIT_FAILURE);
        }
    }

    return ret_val;
}


void deallocate_dynamic_float_matrix(float **matrix, int row)
{
    int i;

    for (i = 0; i < row; ++i)
    {
        free(matrix[i]);
		matrix[i] = NULL;
    }
    free(matrix);
}

void deallocate_dynamic_int_matrix(int **matrix, int row)
{

    int i;

    for (i = 0; i < row; ++i)
    {
        free(matrix[i]);
		matrix[i] = NULL;
    }
    free(matrix);

}



/* uniform distribution, (0..1] */
float drand()   
{
  return (rand()+1.0)/(RAND_MAX+1.0);
}

/* normal distribution, centered on 0, std dev 1 */
float random_normal() 
{
  return sqrt(-2*log(drand())) * cos(2*M_PI*drand());
}

float accuracy(float acc, double *y, double *y_pred, int n)
{
	int idx1 = ArgMax(y_pred, n);
	int idx2 = ArgMax(y, n);

	if (idx1 == idx2)
	{
		acc = acc + 1 ;
	}
	return acc;
}

int ArgMax(double *y, int n)
{
	int indMax = 0;
  double max = y[0];

  for (int i = 1; i < n; i++)
  {
    if (y[i] > max)
	  {
		  indMax = i;
	  }
  }
	return indMax ;
}

float binary_loss_entropy(double *y , double *y_pred, int n) 
{
  float loss;
  int idx;
  idx = ArgMax(y, n);
  loss = -1*log(y_pred[idx]);
  return loss ;
}

float loss_entropy(double *y , double *y_pred, int n) 
{
  float loss = 0.0;
  for (int i = 0; i < n; i++)
  {
    loss =  loss + ( -1*y[i]*log(y_pred[i]) );
  }
  return loss ;
}



void get_split_data(Data *data, float VALIDATION_SIZE)
{
  printf("\n ============= Data Summary ========== \n");

  double a;
	int b ;
  FILE *fin = NULL;
  FILE *file = NULL;
	FILE *stream = NULL;

  fin = fopen("../../../data/data.txt" , "r");
  if(fscanf(fin, "%d" , &data->xraw)){}
  if(fscanf(fin, "%d" , &data->xcol))
  {printf(" *Data shape : (%d , %d) \n" , data->xraw , data->xcol);}

  file = fopen("../../../data/embedding.txt" , "r");
	if(fscanf(file, "%d" , &data->eraw)){}
  if( fscanf(file, "%d" ,&data->ecol))
  {printf(" *Embedding Matrix shape : (%d , %d) \n" , data->eraw , data->ecol);}

  stream = fopen("../../../data/label.txt" , "r");
	if(fscanf(stream,  "%d" , &data->yraw)){}
  if( fscanf(stream, "%d" , &data->ycol)){}

	data->embedding = allocate_dynamic_float_matrix(data->eraw, data->ecol);
	data->X = allocate_dynamic_int_matrix(data->xraw, data->xcol);
	data->Y = allocate_dynamic_float_matrix(data->yraw, data->ycol);

	// Embeddind matrix
	if (file != NULL)
  {
		for (int i = 0; i < data->eraw; i++)
		{
			for (int j = 0; j < data->ecol; j++)
			{
				if(fscanf(file, "%lf" , &a)){
				data->embedding[i][j] = a;
				}
			}
			
		}
  }
	// X matrix
	if (fin != NULL)
  {
		for ( int i = 0; i < data->xraw; i++)
		{
			for ( int j = 0; j < data->xcol; j++)
			{
				if(fscanf(fin, "%d" , &b)){
				data->X[i][j] = b;
				}
			}
		}
  }
	// Y matrix
	if (fin != NULL)
  {
		for ( int i = 0; i < data->yraw; i++)
		{
			for ( int j = 0; j < data->ycol; j++)
			{
				if(fscanf(stream, "%d" , &b))
        {
				data->Y[i][j] = b;
				}
			}
		}
  }

   if (VALIDATION_SIZE == 0)
  {
	  data->start_test = data->xraw * 0.7 ;
    data->start_val = data->start_test;
    data->end_val   = data->xraw - 1;
  }
  else
  {
	  data->start_val = data->xraw * 0.7 ;
	  data->end_val = data->start_val + (data->xraw*VALIDATION_SIZE - 1);
    data->start_test = data->end_val + 1 ;
  }
	printf(" *Train data : 0 ---> %d  \n " , data->start_val - 1);
	printf("*Validation data : %d ---> %d  \n " , (data->start_val), data->end_val);
	printf("*Test data : %d ---> %d \n " , (data->start_test), data->xraw-1);

	fclose(fin);
	fclose(file);
	fclose(stream);
}



