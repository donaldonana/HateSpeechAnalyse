
#ifndef DEF_SIMPLERNN
#define DEF_SIMPLERNN

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


typedef struct DerivedSimpleRNN DerivedSimpleRNN;
struct DerivedSimpleRNN
{
	float **dWhx;
	float **dWhh;
	float **WhhT;
	float *dbh;
	float **dWhy;
	float **WhyT;
	float *dby;
	float *dhraw;
	float *dh;
	float **temp2;
	float **temp3;
};


void training(int epoch, SimpleRNN *rnn, DerivedSimpleRNN *drnn, Data *data, int index) ;

void testing(SimpleRNN *rnn, int **data, int *datadim, float **embedding_matrix, int index, int *target);

void forward(SimpleRNN *rnn, int *x, int n, float **embedding_matrix);

void initialize_rnn(SimpleRNN *rnn, int input_size, int hidden_size, int output_size);

void backforward(SimpleRNN *rnn, int n, int idx, int *x, float **embedding_matrix, 
DerivedSimpleRNN *drnn);

float accuracy(float acc, float y, float *y_pred);

void dhraw(float *dhraw, float *lasth, float *dh, int n);

void deallocate_rnn_derived(SimpleRNN *rnn, DerivedSimpleRNN * drnn);

void initialize_rnn_derived(SimpleRNN *rnn, DerivedSimpleRNN * drnn);

void save_rnn_as_json(SimpleRNN *rnn, FILE *fichier);


#endif
