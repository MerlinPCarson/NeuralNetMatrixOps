import sys
import time
import numpy as np

from model import Model


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

    dimensions = [X_train.shape[1], 500, 2]
    num_epochs = 50000
    paitience = 20 
    lr_rate = 0.01
    model = Model(dimensions, lr_rate, activation='relu')
    model.fit(X_train, y_train, X_val, y_val, num_epochs, paitience=paitience)

    print(f'Script completed in {time.time()-start:.2f} secs')

    return 0

if __name__ == '__main__':
    sys.exit(main())
