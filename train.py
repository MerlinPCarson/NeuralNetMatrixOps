import sys
import time
import numpy as np

from model import Model, one_hot_encode


def one_hot_encode(labels):
    num_labels = len(np.unique(labels))
    encoding = np.eye(num_labels)[labels.astype('uint8')]
    return encoding


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

def main():
    start = time.time()

    train_path = 'data/SPECTF.train'
    val_path = 'data/SPECTF.test'

    X_train, y_train = load_SPECT_data(train_path)
    X_val, y_val = load_SPECT_data(val_path)

    dimensions = [X_train.shape[1], 50, 2]
    num_epochs = 500
    paitience = 10 
    #paitience = None 
    lr_rate = 0.01
    model = Model(dimensions, lr_rate, activation='relu')
    history = model.fit(X_train, y_train, X_val, y_val, num_epochs, paitience=paitience)

    train_acc = model.evaluate(X_train, y_train)
    val_acc = model.evaluate(X_val, y_val)

    print(f'Training accuracy: {train_acc:.4f}, Validation accuracy: {val_acc:.4f}')

    print(f'Script completed in {time.time()-start:.2f} secs')

    return 0

if __name__ == '__main__':
    sys.exit(main())
