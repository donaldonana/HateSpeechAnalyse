#ifndef DEF_UTILS
#define DEF_UTILS

#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))
 

typedef struct SimpleRNN SimpleRNN;
struct SimpleRNN
{
	//self.W_hx = randn(embed_dim, hiden_size)/10
	double **W_hx;
	double **dW_hx;
	//self.W_hh = np.identity(hiden_size)
	double **W_hh;
	double **dW_hh;
	//vecteur de poid couche cachée et couche de sortie (neurons X outputs)
	double *b_h;
	double *db_h;
    // self.W_yh = randn(hiden_size, output_size)/10
	double **W_yh;
	double **dW_yh;
    // self.b_y = np.zeros((output_size, ))
	double *b_y;
	double *db_y;

    double **last_intput;
	double **last_hs;
	double *y ;
	int input_size;
	int hidden_size;
	int output_size;
};



typedef struct DerivedSimpleRNN DerivedSimpleRNN;
struct DerivedSimpleRNN
{
	double **temp2;
	double **temp3;

	double **dWhx;
	double **dWhh;
	double **WhhT;

	double *dbh;
	double **dWhy;
	double **WhyT;
	double *dby;
	double *dhraw;
	double *temp1;
	double *dh;



};


double drand();

double random_normal() ;


double **GetEmbedding(FILE *fin, int row, int col) ;

int **GetData(FILE *fin, int row, int col) ;

double **allocate_dynamic_float_matrix(int row, int col);

int **allocate_dynamic_int_matrix(int row, int col);

void deallocate_dynamic_float_matrix(double **matrix, int row);

void deallocate_dynamic_int_matrix(int **matrix, int row);

void softmax(double *r, int n, double* input);

double binary_loss_entropy(int idx , double *y_pred);

void display_matrix(double **a, int row, int col);

void ToEyeMatrix(double **A, int row, int col) ;

void randomly_initalialize_mat(double **a, int row, int col);

void initialize_rnn(SimpleRNN *rnn, int input_size, int hidden_size, int output_size);

void initialize_mat_zero(double **a, int row, int col);

void initialize_vect_zero(double *a, int n);

void add_vect(double *r , double *a, double *b, int n);

void mat_mul(double *r, double* a, double** b, int n, int p) ;

void forward(SimpleRNN *rnn, int *x, int n, double **embedding_matrix);

void copy_vect(double *a, double *b , int n);

void tan_h(double *r , int n, double* input) ;

int load_target(FILE *stream, int *target );

void backforward(SimpleRNN *rnn, int n, int idx, int *x, double **embedding_matrix, 
DerivedSimpleRNN *drnn);


void vect_mult(double **r, double *a , double *b, int n , int m);

void minus_matrix(double **r, double **a , double **b, int row, int col);

void minus_vect(double *r, double *a, double *b, int n);

void trans_mat(double **r, double **a, int row , int col);

void vect_pow_2(double *r, double *a, int n);

void one_minus_vect(double *r, double *a , int n);

void hadamar_vect(double *r, double *a, double *b, int n);

void add_matrix(double **r, double **a , double **b, int row, int col);

void initialize_rnn_derived(SimpleRNN *rnn, DerivedSimpleRNN * drnn);

void deallocate_rnn_derived(SimpleRNN *rnn, DerivedSimpleRNN * drnn);

#endif
