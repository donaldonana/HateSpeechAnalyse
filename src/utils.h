#ifndef DEF_UTILS
#define DEF_UTILS

#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))
 

typedef struct SimpleRNN SimpleRNN;
struct SimpleRNN
{

	int input_size;
	int hidden_size;
	int output_size;

	//self.W_hx = randn(embed_dim, hiden_size)/10
	float **W_hx;  //Matrice de poids entre la couche cachée et la couche d'entrée de taille m × h
	//self.W_hh = np.identity(hiden_size)
	float **W_hh; //Matrice de poids entre la couche cachée et la couche de contexte de taille h × h
	//vecteur de poids couche cachée et couche de sortie (neurons X outputs)
	float *b_h; //le vecteur de biais entre la couche cachée et la couche d’entrée de taille h
    // self.W_yh = randn(hiden_size, output_size)/10
	float **W_yh;//la matrice de poids entre la couche cachée et la couche de sortie de taille (h × ν)
    // self.b_y = np.zeros((output_size, ))
	float *b_y;//le vecteur de biais entre la couche cachée et la couche de sortie de taille ν

	float **h;//l’état de la couche cachée à l’instant t de taille h
	float *y; //le vecteur finale en sorti de la fonction Sof tM ax taille ν

	float *temp1;
	float *temp2;
	
	
};



typedef struct DerivedSimpleRNN DerivedSimpleRNN;
struct DerivedSimpleRNN
{
	float **temp2;
	float **temp3;
	float **dWhx;
	float **dWhh;
	float **WhhT;
	float *dbh;
	float **dWhy;
	float **WhyT;
	float *dby;
	float *dhraw;
	float *temp1;
	float *dh;
};


float drand();

float random_normal() ;

float **GetEmbedding(int *dim)  ;

int **GetData(int *dim) ; 

float **allocate_dynamic_float_matrix(int row, int col);

int **allocate_dynamic_int_matrix(int row, int col);

void deallocate_dynamic_float_matrix(float **matrix, int row);

void deallocate_dynamic_int_matrix(int **matrix, int row);

void softmax(float *r, int n, float* input);

float binary_loss_entropy(int idx , float *y_pred);

void display_matrix(float **a, int row, int col);

void ToEyeMatrix(float **A, int row, int col) ;

void randomly_initalialize_mat(float **a, int row, int col);

void initialize_rnn(SimpleRNN *rnn, int input_size, int hidden_size, int output_size);

void initialize_mat_zero(float **a, int row, int col);

void initialize_vect_zero(float *a, int n);

void add_vect(float *r , float *a, float *b, int n);

void mat_mul(float *r, float* a, float** b, int n, int p) ;

void forward(SimpleRNN *rnn, int *x, int n, float **embedding_matrix);

void add_vect_three(float *r, float *a, float *b, float *c, int n);

void copy_vect(float *a, float *b , int n);

void tan_h(float *r , int n, float* input) ;

int *load_target(int *target );

void backforward(SimpleRNN *rnn, int n, int idx, int *x, float **embedding_matrix, 
DerivedSimpleRNN *drnn);


void vect_mult(float **r, float *a , float *b, int n , int m);

void minus_matrix(float **r, float **a , float **b, int row, int col);

void minus_vect(float *r, float *a, float *b, int n);

void trans_mat(float **r, float **a, int row , int col);

void vect_pow_2(float *dhraw, float *lasth, float *dh, int n);

void one_minus_vect(float *r, float *a , int n);

void hadamar_vect(float *r, float *a, float *b, int n);

void add_matrix(float **r, float **a , float **b, int row, int col);

void initialize_rnn_derived(SimpleRNN *rnn, DerivedSimpleRNN * drnn);

void deallocate_rnn_derived(SimpleRNN *rnn, DerivedSimpleRNN * drnn);

#endif
