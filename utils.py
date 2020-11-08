import numpy as np

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

def calc_precision(preds, labels):
    # number of true positives
    tp = len(np.where((preds==1) & (labels[:,1]==1))[0])
    # number of positive predictions (true positives + false positives)
    tp_fp = np.sum(preds==1)

    return tp/tp_fp

def calc_recall(preds, labels):
    # number of true positives
    tp = np.sum(len(np.where((preds==1) & (labels[:,1]==1))[0]))
    # number of false negatives
    fn = len(np.where((preds==0) & (labels[:,0]==0))[0])

    return tp/(tp + fn)
    