import numpy as np
from tqdm import tqdm

from q1_softmax import softmax
from q2_sigmoid import sigmoid, sigmoid_grad
from q2_relu import relu, relu_grad
from q2_neural import forward_backward_prop, CE


def one_hot_encode(labels):
    num_labels = len(np.unique(labels))
    encoding = np.eye(num_labels)[labels.astype('uint8')]
    return encoding

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


    def forward(self, data, labels, params, dimensions, activation='sigmoid'):
        """
        Forward propagation for a two-layer sigmoidal or ReLU network

        Compute the forward propagation and for the cross entropy cost,

        Arguments:
        data -- M x Dx matrix, where each row is a training example.
        labels -- M x Dy matrix, where each row is a one-hot vector.
        params -- Model parameters, these are unpacked for you.
        dimensions -- A tuple of input dimension, number of hidden units
                      and output dimension
        """

        # Unpack network parameters (do not modify)
        ofs = 0
        Dx, H, Dy = (dimensions[0], dimensions[1], dimensions[2])

        W1 = np.reshape(params[ofs:ofs + Dx * H], (Dx, H))
        ofs += Dx * H
        b1 = np.reshape(params[ofs:ofs + H], (1, H))
        ofs += H
        W2 = np.reshape(params[ofs:ofs + H * Dy], (H, Dy))
        ofs += H * Dy
        b2 = np.reshape(params[ofs:ofs + Dy], (1, Dy))

        # FOWARD PASS

        # calculate signal at hidden layer 
        z1 = data.dot(W1) + b1

        # calculate ouput of hidden layer 
        if activation == 'sigmoid':
            a1 = sigmoid(z1)
        elif activation == 'relu':
            a1 = relu(z1)

        # calculate signal at output layer
        z2 = a1.dot(W2) + b2
        a2 = softmax(z2)

        # error on from forward pass
        cost = CE(labels, a2)

        return cost, a2

    def backward(self):
        pass

    def evaluate(self, X, y):

        # get model predictions
        _, outputs = self.forward(X, y, self.weights, self.dimensions, self.activation)
        preds = np.argmax(outputs, axis=1)

        # determine accuracy of predictions
        preds_enc = one_hot_encode(preds)
        compare = [preds_enc[i,0]==y[i,0] and preds_enc[i,1]==y[i,1] for i in range(y.shape[0])]
        accuracy = np.sum(compare)/outputs.shape[0]
        return accuracy

    def fit(self, X_train, y_train, X_val, y_val, num_epochs, paitience=None):

        history = {'loss': [], 'val_loss': []}
        best_val_loss = 999
        best_epoch = 0
        epochs_since_val_decrease = 0
        order = np.arange(0, X_train.shape[0], 1)
        for epoch in range(num_epochs):
            np.random.shuffle(order)
            loss, _ = forward_backward_prop(X_train[order], y_train[order], self.weights, self.dimensions, self.activation)
            val_loss, _ = self.forward(X_val, y_val, self.weights, self.dimensions, self.activation)
            history['loss'].append(loss)
            history['val_loss'].append(val_loss)
            print(f'Epoch[{epoch+1}/{num_epochs}]: loss = {loss}, val loss = {val_loss}')

            # save weights with lowest validation loss
            if best_val_loss > val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                epochs_since_val_decrease = 0
                self.best_weights = self.weights

            # check for early stopping
            if paitience is not None:
                epochs_since_val_decrease += 1
                if epochs_since_val_decrease >= paitience:
                    print(f'Validation has not decreased in {epochs_since_val_decrease}: triggering early stopping.')
                    break
        
        print(f'Training complete: lowest validation was {best_val_loss} at epoch {best_epoch}')
        return history

