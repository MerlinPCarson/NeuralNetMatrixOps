import sys
import time
import numpy as np
import matplotlib.pyplot as plt

from model import Model, one_hot_encode


def load_SPECT_data(path):
    data = np.genfromtxt(path, delimiter=',')

    # turn labels into one hot encoding
    labels = one_hot_encode(data[:,0])

    # get features without labels
    features = data[:,1:]

    # standardize data
    feat_mean = np.mean(features)
    feat_std = np.std(features)
    features = (features-feat_mean)/feat_std

    return features, labels

def plot_history(history):

    epochs = np.arange(1, len(history['loss'])+1, 1)

    plt.plot(epochs, history['loss'], label='Training')
    plt.plot(epochs, history['val_loss'], label='Validation')

    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

def main():
    start = time.time()

    # seed number generator for experiment reproducability
    np.random.seed(42)

    train_path = 'data/SPECTF.train'
    val_path = 'data/SPECTF.test'

    X_train, y_train = load_SPECT_data(train_path)
    X_val, y_val = load_SPECT_data(val_path)

    dimensions = [X_train.shape[1], 50, 2]
    num_epochs = 100 
    paitience = 10 
    paitience = None 
    lr = 0.01
    mb_size = 16 
    model = Model(dimensions, lr, activation='sigmoid')
    history = model.fit(X_train, y_train, X_val, y_val, num_epochs, mb_size, paitience=paitience)

    # determine accuracies for data sets
    train_acc, train_metrics = model.evaluate(X_train, y_train)
    val_acc, val_metrics = model.evaluate(X_val, y_val)
    print(f'Training accuracy:   {train_acc:.4f}, precision: {train_metrics["precision"]:.4f},', \
          f'recall: {train_metrics["recall"]:.4f}, F1: {train_metrics["f1"]:.4f}')
    print(f'Validation accuracy: {val_acc:.4f}, precision: {val_metrics["precision"]:.4f},', \
          f'recall: {val_metrics["recall"]:.4f}, f1: {val_metrics["f1"]:.4f}')

    plot_history(history)

    print(f'Script completed in {time.time()-start:.2f} secs')

    return 0

if __name__ == '__main__':
    sys.exit(main())
