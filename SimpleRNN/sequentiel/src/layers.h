#include <stdlib.h>
#include <math.h>

/*! \file layers.h
    \brief Various mathematical functions
    
    An LSTM network is built out of a series
    of these type of mathematical operations.
*/

// Dealing with FC layers, forward and backward
/**	Y = AX + b 
*
*  A(rows: R, columns: C)
*/
void fully_connected_forward(double* Y, double* A, double* X,
	double* b, int R, int C);
/**		Y = AX + b
* 
* A(rows: R, columns: C)
*
* dld* points to gradients
*/
void fully_connected_backward(double* dldY, double* A, double* X,double* dldA,
  double* dldX, double* dldb, int R, int C);

/** Softmax layer forward propagation
*
* @param P sum ( exp(y / \pa temperature) ) for y in \pa Y
* @param Y input 
* @param temperature calibration of softmax, the lower the spikier
* @param F len ( Y )  
*/
void softmax_layers_forward(double* P, double* Y, int F);
/** Softmax layer backward propagation
*
* @param P sum ( exp(y/temperature) ) for y in Y
* @param c correct prediction
* @param dldh gradients back to Y, given \p c
* @param F len ( Y )  
*/
void softmax_loss_layer_backward(double* P, int c, double* dldh, int F);

// Other layers used: sigmoid and tanh
// 	
/** Y = sigmoid(X)
*
* L = len(X) 
*/
void sigmoid_forward(double* Y, double* X, int L);
/** Y = sigmoid(X), dldY, Y, &dldX, length */
void sigmoid_backward(double* dldY, double* Y, double* dldX, int L);
/** Y = tanh(X), &Y, X, length */
void tanh_forward(double* Y, double* X, int L);
/** Y = tanh(X), dldY, Y, &dldX, length */
void tanh_backward(double* dldY, double* Y, double* dldX, int L);

/** The loss function used in the output layer of the LSTM network, which is a softmax layer 
* \see softmax_layers_forward
* @param probabilities array with output from \ref softmax_layers_forward 
* @param correct the index that represents the correct observation
*/
double cross_entropy(double* probabilities, int correct);
