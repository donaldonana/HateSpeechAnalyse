
#ifndef DEF_SIMPLERNN
#define DEF_SIMPLERNN

#include "algebre.h"
#include "utils.h"

 

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
};


typedef struct gradient gradient;
struct gradient
{
	float **dWhx;
	float **dWhh;
	float **WhhT;
	float **dWhy;
	float *dbh;
	float *dby;
	float *dh;
	float *dhraw;
	// Necessary Cache
	float **WhyT;
	float **temp2;
	float **temp3;
};


void training(int epoch, SimpleRNN *rnn, gradient *drnn, SimpleRNN *AVGgradient, Data *data, int index, int mini_batch) ;

void testing(SimpleRNN *rnn, int **data, int *datadim, float **embedding_matrix, int index, int *target);

void forward(SimpleRNN *rnn, int *x, int n, float **embedding_matrix);

void initialize_rnn(SimpleRNN *rnn, int input_size, int hidden_size, int output_size);

void backforward(SimpleRNN *rnn, int n, int idx, int *x, float **embedding_matrix, gradient *drnn, SimpleRNN *AVGgradient);

float accuracy(float acc, float y, float *y_pred);

void print_summary(SimpleRNN* rnn, int epoch, int mini_batch, float lr);

void dhraw(float *dhraw, float *lasth, float *dh, int n);

void deallocate_rnn_derived(SimpleRNN *rnn, gradient* drnn);

void initialize_rnn_derived(SimpleRNN *rnn, gradient* drnn);

void deallocate_rnn_gradient(SimpleRNN *AVGgradient);

void zero_rnn_gradient(SimpleRNN *AVGgradient);

void gradient_descent(SimpleRNN *rnn, SimpleRNN *AVGgradient, int n, float lr);

void save_rnn_as_json(SimpleRNN *rnn, FILE *fichier);

#endif
