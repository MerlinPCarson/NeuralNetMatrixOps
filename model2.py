import numpy as np
from tqdm import tqdm
from math import ceil

from q1_softmax import softmax
from q2_sigmoid import sigmoid, sigmoid_grad
from q2_relu import relu, relu_grad
from q2_neural_one_more import forward_backward_prop, CE

from utils import calc_precision, calc_recall

class Model2():

    def __init__(self, dimensions, lr, activation='sigmoid'):

        self.dimensions = dimensions
        self.lr = lr
        self.activation = activation
        self.glorot_weight_init(self.dimensions)
        
    def glorot_weight_init(self, dimensions):
        weights = [np.random.normal(0, np.sqrt(2/(in_dim+1+out_dim)), size=(in_dim+1,out_dim)) 
                       for in_dim, out_dim in zip(dimensions[:-1], dimensions[1:])]

        self.weights = weights[0].flatten()
        for layer in range(1, len(weights)):
            self.weights = np.concatenate((self.weights, weights[layer].flatten()))

    def forward(self, data, labels, params, dimensions, activation='sigmoid'):
        """
        Forward propagation for a three-layer sigmoidal or ReLU network
    
        Compute the forward propagation and for the cross entropy cost,
        and backward propagation for the gradients for all parameters.
    
        Arguments:
        data -- M x Dx matrix, where each row is a training example.
        labels -- M x Dy matrix, where each row is a one-hot vector.
        params -- Model parameters, these are unpacked for you.
        dimensions -- A tuple of input dimension, number of hidden units
                      and output dimension
        """
    
        # Unpack network parameters (do not modify)
        ofs = 0
        Dx, H1, H2, Dy = (dimensions[0], dimensions[1], dimensions[2], dimensions[3])
    
        W1 = np.reshape(params[ofs:ofs + Dx * H1], (Dx, H1))
        ofs += Dx * H1
        b1 = np.reshape(params[ofs:ofs + H1], (1, H1))
        ofs += H1
        W2 = np.reshape(params[ofs:ofs + H1 * H2], (H1, H2))
        ofs += H1 * H2 
        b2 = np.reshape(params[ofs:ofs + H2], (1, H2))
        ofs += H2
        W3 = np.reshape(params[ofs:ofs + H2 * Dy], (H2, Dy))
        ofs += H2 * Dy
        b3 = np.reshape(params[ofs:ofs + Dy], (1, Dy))
    
        # FOWARD PASS
    
        # calculate signal at 1st hidden layer 
        z1 = data.dot(W1) + b1
    
        # calculate ouput of 1st hidden layer 
        if activation == 'sigmoid':
            a1 = sigmoid(z1)
        elif activation == 'relu':
            a1 = relu(z1)
    
        # calculate signal at 2nd hidden layer layer
        z2 = a1.dot(W2) + b2
    
        # calculate ouput of 2nd hidden layer 
        if activation == 'sigmoid':
            a2 = sigmoid(z2)
        elif activation == 'relu':
            a2 = relu(z2)
    
        # calculate signal at output layer
        z3 = a2.dot(W3) + b3
        a3 = softmax(z3)
    
        # error on from forward pass
        cost = CE(labels, a3)
    
        return cost, a3

    def evaluate(self, X, y):

        # get model predictions
        _, outputs = self.forward(X, y, self.best_weights, self.dimensions, self.activation)
        preds = np.argmax(outputs, axis=1)

        precision = calc_precision(preds, y)
        recall = calc_recall(preds, y)
        f1 = 2*(precision*recall)/(precision+recall)

        # determine accuracy of predictions
        compare = [y[i, pred]==1 for i, pred in enumerate(preds)]
        accuracy = np.sum(compare)/outputs.shape[0]
        return accuracy, {'precision': precision, 'recall': recall, 'f1':f1}

    def fit(self, X_train, y_train, X_val, y_val, num_epochs, mb_size, paitience=None):

        best_val_loss = 999
        best_epoch = 0
        epochs_since_val_decrease = 0
        order = np.arange(0, X_train.shape[0], 1)
        num_batches = ceil(X_train.shape[0]/mb_size)

        # epoch 0 losses, no trainning
        loss, _ = self.forward(X_train, y_train, self.weights, self.dimensions, self.activation)
        val_loss, _ = self.forward(X_val, y_val, self.weights, self.dimensions, self.activation)
        history = {'loss': [loss], 'val_loss': [val_loss]}

        # main training loop
        for epoch in range(num_epochs):
            epoch_loss = 0.0

            # shuffle training data 
            np.random.shuffle(order)
            X_train = X_train[order]
            y_train = y_train[order]

            # train with mini batches
            for mb in range(num_batches):
                mb_start = mb * mb_size
                mb_end = mb_start + mb_size
                loss, grad = forward_backward_prop(X_train[mb_start:mb_end], y_train[mb_start:mb_end], self.weights, self.dimensions, self.activation)
                self.weights -= self.lr * grad
                epoch_loss += loss

            # process validation examples 
            val_loss, _ = self.forward(X_val, y_val, self.weights, self.dimensions, self.activation)

            # save results
            epoch_loss /= num_batches
            history['loss'].append(epoch_loss)
            history['val_loss'].append(val_loss)
            print(f'Epoch[{epoch+1}/{num_epochs}]: loss = {epoch_loss}, val loss = {val_loss}')

            # save weights with lowest validation loss
            if best_val_loss > val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                epochs_since_val_decrease = 0
                self.best_weights = np.copy(self.weights)

            # check for early stopping
            if paitience is not None:
                epochs_since_val_decrease += 1
                if epochs_since_val_decrease >= paitience:
                    print(f'Validation has not decreased in {epochs_since_val_decrease}: triggering early stopping.')
                    break
        
        print(f'Training complete: lowest validation was {best_val_loss} at epoch {best_epoch+1}')
        return history

