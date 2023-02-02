

#ifndef DEF_UTILS
#define DEF_UTILS



#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))
 

float drand();

float random_normal() ;

float rounded_float(float val);

float **allocate_dynamic_float_matrix(int row, int col);

int **allocate_dynamic_int_matrix(int row, int col);

void deallocate_dynamic_float_matrix(float **matrix, int row);

void deallocate_dynamic_int_matrix(int **matrix, int row);

void softmax(float *r, float* input, int n);

float binary_loss_entropy(int idx , float *y_pred);

void display_matrix(float **a, int row, int col);

void ToEyeMatrix(float **A, int row, int col) ;

void randomly_initalialize_mat(float **a, int row, int col);

void initialize_mat_zero(float **a, int row, int col);

void initialize_vect_zero(float *a, int n);

void add_vect(float *r , float *a, float *b, int n);

void mat_mul(float *r, float* a, float** b, int n, int p) ;

void add_three_vect(float *r, float *a, float *b, float *c, int n);

void copy_vect(float *a, float *b , int n);

void Tanh(float *r , float* input, int n) ;

void vect_mult(float **r, float *a , float *b, int n , int m);

void minus_matrix(float **r, float **a , float **b, int row, int col);

void minus_vect(float *r, float *a, float *b, int n);

void trans_mat(float **r, float **a, int row , int col);

void add_matrix(float **r, float **a , float **b, int row, int col);

void update_matrix(float **r, float **a , float **b, int row, int col, int n, float lr);

void update_vect(float *r, float *a, float *b, int col , int n, float lr);

int ArgMax( float *y_pred);

#endif
