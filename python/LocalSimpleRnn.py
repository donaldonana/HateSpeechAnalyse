 
import os
import numpy as np
import pandas as pd
from numpy.random import randn







# Utily Fonction
    
def shuffle_split_data(X, y):
    split = np.random.rand(X.shape[0]) < 0.7

    X_Train = X[split]
    y_Train = y[split]
    X_Test =  X[~split]
    y_Test = y[~split]

    return X_Train, y_Train, X_Test, y_Test


def get_minibatch(X, y, minibatch_size, shuffle=True):
  minibatches = []
  for i in range(0, X.shape[0], minibatch_size):
    X_mini = X[i:i + minibatch_size]
    y_mini = y[i:i + minibatch_size]
    minibatches.append((X_mini, y_mini))
  return minibatches


def get_embedding_vector(x, embedding_matrix, embed_dim):
  sentences = []
  embedd_vector = np.array((1,embed_dim))
  for xt in x:
    sentences.append(embedding_matrix[xt])
  return sentences


def softmax( x, axis = -1):
      """Compute softmax values for each sets of scores in x."""
      return np.exp(x) / np.sum(np.exp(x), axis=axis)

def binary_loss_entropy(label, y_pred):
      """Compute the loss entropy beetween between two binary probability distributions"""
      idx = np.argmax(label)
      loss = -np.log(y_pred[idx])
      return loss

def accuracy(n, y, y_pred):
      target = np.argmax(y[0])
      n += int(np.argmax(y_pred[0]) == target)
      return  n



file_name = "embedding.txt"

count = 0
embedding_matrix = []

print(" \n word embedding in process ... \n ")
with open(file_name) as infile:
	for line in infile:
		if count == 0:
			a = line.strip()
			shape = list(map(int, a.split(" ")))
		else :
			a = line.strip()
			# print(a)
			vect = list(map(float, a.split(" ")))
			embedding_matrix.append(vect)

		count = count + 1

embedding_matrix = np.array(embedding_matrix)
print(" Word embedding end embedding matrix shape is : ", embedding_matrix.shape)




print(" \n Reading data in process ... \n ")
url = "DynamicallyGeneratedHateDataset.csv"
df = pd.read_csv(url)
data = df[['text','label']]
data = data[data.label != "Neutral"]
Y = pd.get_dummies(data.label).values

X = []
count = 0
with open("data.txt") as infile:
  for line in infile:
  	if count == 0:
  		a = line.strip()
  		shape = list(map(int, a.split(" ")))
  	else :
  		a = line.strip()
  		# print(a)
  		v = list(map(int, a.split(" ")))
  		X.append(v)
  	count = count + 1

X = np.array(X)

try:
  X_train, Y_train, X_test, Y_test = shuffle_split_data(X,Y)
#     print("Successful shuffle_split")
except:
  print("shuffle_split Fail")

print(" ---- Reading data end ---- \n")
print(" Train Shape : ")
print(X_train.shape,Y_train.shape)
print(" \n Test Shape : ")
print(X_test.shape,Y_test.shape)





hiden_size = 64
embed_dim = 128 
output_size = 2
class MyRnn():
    
    def __init__(self, num_units=64, input_size=128, output_size=2):
        super(MyRnn, self).__init__()
        self.num_units = num_units
        self.input_size = input_size
        self.output_size = output_size
 
    def build(self):
          
        # self.W_hx = Wxh.numpy() 
        self.W_hx = randn(embed_dim, hiden_size)/10

        # self.W_hh = Whh.numpy()
        self.W_hh = np.identity(hiden_size)

        # self.b_h = bh.numpy()
        self.b_h = np.zeros((hiden_size, ))

        # self.W_yh = Why.numpy()
        self.W_yh = randn(hiden_size, output_size)/10

        # self.b_y = by.numpy()
        self.b_y = np.zeros((output_size, ))

        self.weights = [ self.W_hx, self.W_hh, self.b_h, self.W_yh,  self.b_h,  self.b_y]
        
        self.dW_hx = np.zeros(self.W_hx.shape)
        self.dW_hh = np.zeros(self.W_hh.shape)
        self.db_h = np.zeros(self.b_h.shape)
        self.dW_yh = np.zeros(self.W_yh.shape)
        self.db_y = np.zeros(self.b_y.shape)

        self.last_input = None

        self.h_last = None

        #self.nbr_correct = 0
    



    def forward(self, sequence):
        """ Like the forward phase """
        h = [] 
        self.h_last = np.zeros([1, self.num_units])
        h.append(self.h_last)
        x = get_embedding_vector(sequence, embedding_matrix, 128)
        self.last_input = x

        for xt in x:
            xt = np.expand_dims(xt, axis=0)
            h_t =  np.dot(xt, self.W_hx)  +  np.dot(self.h_last, self.W_hh)  + self.b_h     
            self.h_last = np.tanh(h_t)
            h.append(self.h_last)

        y = np.dot(self.h_last, self.W_yh) + self.b_y

        return  softmax(y) , h
    
    def back_forward(self, X, Y, y_hat, h, learning_rate=0.01): 
             
        loss = binary_loss_entropy(Y, y_hat[0])

         # dy = y_pred - label
        dy = y_hat - Y
         
        #dWhy = last_h.T * dy 
        dWhy = np.dot( self.h_last.T , dy)

        #dby = dy = y_pred - label
        dby = dy[0]

        # Firstly we Initialize dWhh, dWhx, and dbh to zero 
        dWhh =  np.zeros(self.W_hh.shape)
        dWhx =  np.zeros(self.W_hx.shape)
        dbh =  np.zeros(self.b_h.shape)

 
        # dWhh, dWhx, and dbh was computed over time via back propagation through time

        dh = np.matmul( dy , self.W_yh.T  )



        for t in reversed(range(len(X))):

          # dh = np.dot(Why.T, dy) + dhnext ----- à revoir au cas où ------


          #dhraw = (1 - hs[t] * hs[t]) * dh
              dhraw = (1 - np.power( h[t+1], 2 ))  * dh   
  
          # dbh += dhraw
              dbh += dhraw[0]

          # dWxh += np.dot(dhraw, xs[t].T)
              dWhx += (np.matmul(np.expand_dims(self.last_input[t], axis=1) , dhraw))   
          #dWhh += np.dot(dhraw, hs[t-1].T)
              dWhh += (np.matmul( h[t].T, dhraw)) 
          #dhnext = np.dot(Whh.T, dhraw)
              dh = np.matmul( dhraw, self.W_hh.T )

      # Just aggregate the derived weight for this batch
    
        derived = [dWhx*0.5, dWhh*0.5, dbh*0.5, dWhy*0.5, dby*0.5]

        self.dW_hx = self.dW_hx + dWhx*0.5
        self.dW_hh = self.dW_hh + dWhh*0.5
        self.db_h =  self.db_h  + dbh*0.5
        self.dW_yh = self.dW_yh + dWhy*0.5
        self.db_y =  self.db_y  + dby*0.5

        return loss,  derived
    
    

    def gradient(self,  n, learning_rate=0.01):
        
        "Update the parameter with SGD"
        self.W_hx = self.W_hx - learning_rate*self.dW_hx*(1/n)
        self.W_hh = self.W_hh - learning_rate*self.dW_hh*(1/n)
        self.W_yh = self.W_yh - learning_rate*self.dW_yh*(1/n)
        self.b_h =  self.b_h  - learning_rate*self.db_h*(1/n)
        self.b_y =  self.b_y  - learning_rate*self.db_y*(1/n)
        "Gradient update is finish for this batch we need to reinitialize the derived for other batch"
        self.dW_hx = np.zeros(self.W_hx.shape)
        self.dW_hh = np.zeros(self.W_hh.shape)
        self.db_h  = np.zeros(self.b_h.shape)
        self.dW_yh = np.zeros(self.W_yh.shape)
        self.db_y  = np.zeros(self.b_y.shape)



    
    def evaluate(self, x_test, y_test):
        
        Loss = 0
        y_pred = []
        
        for x in x_test:
            y , _ = self.forward(x_mb)
            y_pred.append(y)
            
        for (y_true, y_hat) in zip(y_test, y_pred):
            
            Loss = Loss + binary_loss_entropy(y_true, y_hat[0])
            
        return  (round(Loss/len(x_test), 4))



            
batches = get_minibatch(X[:4000], Y[:4000] ,1)           
print("\n TRAINING PHASE IN PROCESS ... ")

myrnn = MyRnn()
myrnn.build()
epoch = 200
for iter in range(20):
    print("\n Start of epoch %d/200" % (iter+1,))
    Loss = 0
    for (x_mb, y_mb) in batches:
        loss = 0
        n = 0
        for a , b in zip(x_mb, y_mb):
            y_pred, h = myrnn.forward(a)
            l, g = myrnn.back_forward(a, b, y_pred, h)
            loss = loss + l
            n = n + 1
        loss = loss / n
        myrnn.gradient(n)

        Loss = Loss + loss
    print("Loss : ",   round(Loss/len(batches), 4)  )
#     print("Loss : ",   round(loss/len(batches), 4) , " ||  Accuracy : ", round(acc/len(batches), 4)  )


            
    

