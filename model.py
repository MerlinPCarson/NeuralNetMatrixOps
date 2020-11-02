import numpy as np
from tqdm import tqdm

from q1_softmax import softmax
from q2_sigmoid import sigmoid, sigmoid_grad
from q2_relu import relu, relu_grad
from q2_neural import forward_backward_prop
    
class ProgressBar(tqdm):

    def update_progress(self, block_num=1, block_size=1, total_size=None):
        self.update('666')  # will also set self.n = b * bsize

class Model():

    def __init__(self, dimensions, lr_rate, activation='sigmoid'):

        self.dimensions = dimensions
        self.lr_rate = lr_rate
        self.activation = activation
        self.glorot_weight_init(self.dimensions)
        
    def glorot_weight_init(self, dimensions):
        weights = [np.random.normal(0, np.sqrt(2/(in_dim+1+out_dim)), size=(in_dim+1,out_dim)) 
                       for in_dim, out_dim in zip(dimensions[:-1], dimensions[1:])]

        self.weights = weights[0].flatten()
        for layer in range(1, len(weights)):
            self.weights = np.concatenate((self.weights, weights[layer].flatten()))


    def forward(self):
        pass

    def backward(self):
        pass

    def fit(self, X_train, y_train, X_val, y_val, num_epochs, paitience=None):

        best_loss = 999
        epochs_since_val_decrease = 0
        for epoch in range(num_epochs):
            loss, _ = forward_backward_prop(X_train, y_train, self.weights, self.dimensions, self.activation)
            val_loss, _ = forward_backward_prop(X_val, y_val, self.weights, self.dimensions, self.activation)
            print(f'Epoch[{epoch+1}/{num_epochs}]: loss = {loss}, val loss = {val_loss}')

            if paitience is not None:
                # check for early stopping
                if best_loss > val_loss:
                    best_loss = val_loss
                    epochs_since_val_decrease = 0
                    self.best_weights = self.weights
                else:
                    epochs_since_val_decrease += 1
                    if epochs_since_val_decrease >= paitience:
                        print(f'Validation has not decreased in {epochs_since_val_decrease}: triggering early stopping.')
                        break
        
            

