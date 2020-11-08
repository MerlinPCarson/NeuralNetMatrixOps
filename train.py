import sys
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt

from utils import one_hot_encode
from model import Model
from model2 import Model2
from utils import load_SPECT_data 


def plot_history(history):

    plt.plot(history['loss'], label='Training')
    plt.plot(history['val_loss'], label='Validation')

    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

def parse_args():
    parser = argparse. ArgumentParser(description='Gamma-Spectra Denoising Trainer')
    parser.add_argument('--train_path', type=str, default='data/SPECTF.train', help='SPECTF training data')
    parser.add_argument('--test_path', type=str, default='data/SPECTF.test', help='SPECTF testing data')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size for training')
    parser.add_argument('--num_epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--patience', type=int, default=None, help='number of epochs of no improvment before early stopping')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--activation', type=str, default='sigmoid', choices=['sigmoid', 'relu'], help='activation function')
    parser.add_argument('--num_layers', type=int, default=2, choices=[1,2], help='number of hidden layers')
    parser.add_argument('--num_neurons1', type=int, default=50, help='number of neurons in 1st hidden layer')
    parser.add_argument('--num_neurons2', type=int, default=50, help='number of neurons in 2nd hidden layer')
    parser.add_argument('--seed', type=int, default=42, help='random seed')

    return parser.parse_args()

def main(args):
    start = time.time()

    # seed number generator for experiment reproducability
    np.random.seed(args.seed)

    X_train, y_train = load_SPECT_data(args.train_path)
    X_test, y_test = load_SPECT_data(args.test_path)

    # create balanced validation set from test set 
    X_val = X_test #[-30:,:]
    y_val = y_test #[-30:]

    # instanciate model
    if args.num_layers == 1:
        dimensions = [X_train.shape[1], args.num_neurons1, 2]
        model = Model(dimensions, args.lr, activation=args.activation)
    elif args.num_layers == 2:
        dimensions = [X_train.shape[1], args.num_neurons1, args.num_neurons2, 2]
        model = Model2(dimensions, args.lr, activation=args.activation)

    # train model
    history = model.fit(X_train, y_train, X_val, y_val, args.num_epochs, args.batch_size, args.patience)

    # determine accuracies for data sets
    train_acc, train_metrics = model.evaluate(X_train, y_train)
    val_acc, val_metrics = model.evaluate(X_val, y_val)
    test_acc, test_metrics = model.evaluate(X_test, y_test)
    print(f'Training accuracy:   {train_acc:.4f}, precision: {train_metrics["precision"]:.4f},', \
          f'recall: {train_metrics["recall"]:.4f}, F1: {train_metrics["f1"]:.4f}')
    print(f'Validation accuracy: {val_acc:.4f}, precision: {val_metrics["precision"]:.4f},', \
          f'recall: {val_metrics["recall"]:.4f}, f1: {val_metrics["f1"]:.4f}')
    print(f'Test accuracy: {test_acc:.4f}, precision: {test_metrics["precision"]:.4f},', \
          f'recall: {test_metrics["recall"]:.4f}, f1: {test_metrics["f1"]:.4f}')

    plot_history(history)

    print(f'Script completed in {time.time()-start:.2f} secs')

    return 0

    history = model.fit(X_train, y_train, X_val, y_val, args.num_epochs, args.mb_size, args.paitience)

    # determine accuracies for data sets
    train_acc, train_metrics = model.evaluate(X_train, y_train)
    val_acc, val_metrics = model.evaluate(X_val, y_val)
    test_acc, test_metrics = model.evaluate(X_test, y_test)
    print(f'Training accuracy:   {train_acc:.4f}, precision: {train_metrics["precision"]:.4f},', \
          f'recall: {train_metrics["recall"]:.4f}, F1: {train_metrics["f1"]:.4f}')
    print(f'Validation accuracy: {val_acc:.4f}, precision: {val_metrics["precision"]:.4f},', \
          f'recall: {val_metrics["recall"]:.4f}, f1: {val_metrics["f1"]:.4f}')
    print(f'Test accuracy: {test_acc:.4f}, precision: {test_metrics["precision"]:.4f},', \
          f'recall: {test_metrics["recall"]:.4f}, f1: {test_metrics["f1"]:.4f}')

    plot_history(history)

    print(f'Script completed in {time.time()-start:.2f} secs')

    return 0

if __name__ == '__main__':
    sys.exit(main(parse_args()))
