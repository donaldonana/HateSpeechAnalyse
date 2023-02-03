

#ifndef DEF_UTILS
#define DEF_UTILS



#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))
 


typedef struct Data Data;
struct Data
{
	int xraw;
	int xcol;
	int ecol;
    int eraw;
	int start_val;
	int end_val;
	int **X;
    int *Y;
    float **embedding;

};




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

int *load_target(int *target );

void vect_mult(float **r, float *a , float *b, int n , int m);

void update_matrix(float **r, float **a , float **b, int row, int col, int n, float lr);

void update_vect(float *r, float *a, float *b, int col , int n, float lr);

void trans_mat(float **r, float **a, int row , int col);

void data_for_plot(char *filename, int epoch, float *axis, char *axis_name);

void add_matrix(float **r, float **a , float **b, int row, int col);

void vector_store_as_json(float *r, int n, FILE *fo);

void matrix_strore_as_json(float **m, int row, int col, FILE *fo);

void copy_mat(float **a, float **b, int row ,int col);

int ArgMax( float *y_pred);

void update_matrix_model( float **a , float **b, int row, int col, int n);

void update_vect_model(float *a, float *b, int col , int n);

void get_data(Data *data, int nthread);


#endif