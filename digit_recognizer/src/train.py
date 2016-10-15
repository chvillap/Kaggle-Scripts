# =============================================================================
# train.py
#
# Script with functions for training and evaluating a classifier.
# =============================================================================

import numpy as np
import sys
from nolearn.lasagne import BatchIterator
from scipy.ndimage.interpolation import rotate


class RotateBatchIterator(BatchIterator):
    """Performs data augmentation by rotating half of the images in the batch.
    """
    def transform(self, Xb, yb):
        Xb, yb = super(RotateBatchIterator, self).transform(Xb, yb)
        batch_size = Xb.shape[0]
        indices = np.random.choice(batch_size, batch_size / 2, replace=False)
        ind_pos = indices[0::2]
        ind_neg = indices[1::2]
        Xb[ind_pos] = rotate(Xb[ind_pos], angle=30.0, axes=(3, 2),
            reshape=False, order=1)
        Xb[ind_neg] = rotate(Xb[ind_neg], angle=-30.0, axes=(3, 2),
            reshape=False, order=1)
        return Xb, yb


class AdjustParameter(object):
    """Adjusts the value of some parameter of the network at the end of each
    epoch, in order to make it vary over a range of predefined values.
    """
    def __init__(self, name, start, stop):
        self.name = name
        self.start = start
        self.stop = stop
        self.ls = None

    def __call__(self, nn, train_history):
        if self.ls is None:
            self.ls = np.linspace(self.start, self.stop, nn.max_epochs)

        epoch = train_history[-1]['epoch']
        new_value = np.cast['float32'](self.ls[epoch - 1])
        getattr(nn, self.name).set_value(new_value)


class EarlyStopping(object):
    """Stops the learning process early if the network spends too many epochs
    without any performance improvement.
    """
    def __init__(self, patience):
        self.patience = patience
        self.best_valid = np.Inf
        self.best_valid_epoch = 0
        self.best_weights = None

    def __call__(self, nn, train_history):
        current_valid = train_history[-1]['valid_loss']
        current_epoch = train_history[-1]['epoch']

        if current_valid < self.best_valid:
            self.best_valid = current_valid
            self.best_valid_epoch = current_epoch
            self.best_weights = nn.get_all_params_values()

        elif self.best_valid_epoch + self.patience < current_epoch:
            print('Early stopping')
            print('Best valid loss was %f at epoch %d' %
                  (self.best_valid, self.best_valid_epoch))
            nn.load_params_from(self.best_weights)
            raise StopIteration()


def train(X, y):
    """Trains a ConvNet classifier on some training data.

    Inputs:
        X  Training data.
        y  Training labels.

    Outputs:
        clf  Trained classifier.
    """
    from theano import shared
    from nolearn.lasagne import NeuralNet
    from nolearn.lasagne import TrainSplit
    from lasagne.layers import InputLayer
    from lasagne.layers import Conv2DLayer
    from lasagne.layers import DenseLayer
    from lasagne.layers import MaxPool2DLayer
    from lasagne.layers import DropoutLayer
    from lasagne.nonlinearities import softmax

    layers = [
        (InputLayer, {'shape': (None, X.shape[1], X.shape[2], X.shape[3])}),
        (Conv2DLayer, {'num_filters': 32, 'filter_size': 3, 'pad': 'same'}),
        (MaxPool2DLayer, {'pool_size': 2}),
        (DropoutLayer, {'p': 0.1}),
        (Conv2DLayer, {'num_filters': 64, 'filter_size': 3, 'pad': 'same'}),
        (MaxPool2DLayer, {'pool_size': 2}),
        (DropoutLayer, {'p': 0.2}),
        (Conv2DLayer, {'num_filters': 128, 'filter_size': 3, 'pad': 'same'}),
        (MaxPool2DLayer, {'pool_size': 2}),
        (DropoutLayer, {'p': 0.3}),
        (DenseLayer, {'num_units': 512}),
        (DropoutLayer, {'p': 0.4}),
        (DenseLayer, {'num_units': 256}),
        (DropoutLayer, {'p': 0.5}),
        (DenseLayer, {'num_units': 10, 'nonlinearity': softmax}),
    ]
    clf = NeuralNet(
        layers,
        max_epochs=100,
        train_split=TrainSplit(eval_size=0.25),
        # batch_iterator_train=RotateBatchIterator(batch_size=128),
        on_epoch_finished=[
            AdjustParameter('update_learning_rate', start=0.01, stop=0.0001),
            AdjustParameter('update_momentum', start=0.9, stop=0.999),
            EarlyStopping(patience=10),
        ],
        update_learning_rate=shared(np.cast['float32'](0.01)),
        update_momentum=shared(np.cast['float32'](0.9)),
        verbose=2,
    )

    return clf.fit(X, y)


def evaluate(clf, X, y):
    """Evaluates a pretrained classifier on some test data.

    Inputs:
        clf  Pretrained classifier.
        X    Test data.
        y    Test labels.

    Outputs:
        y_pred  Array of predicted labels.
    """
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import classification_report

    y_pred = clf.predict(X)

    print('ACCURACY:')
    print(((y == y_pred).sum() / y.size))
    print()

    print('CONFUSION MATRIX:')
    print(confusion_matrix(y, y_pred))
    print()

    print('CLASSIFICATION REPORT:')
    print(classification_report(y, y_pred))
    print()

    return y_pred


def persist(clf, name):
    """Persists a pretrained classifier into a set of files.

    Inputs:
        clf   Pretrained classifier.
        name  Classifier name.

    Outputs:
        (None)
    """
    import pickle

    sys.setrecursionlimit(10000)
    with open('%s.pickle' % name, 'wb') as f:
        pickle.dump(clf, f, -1)


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: %s dataset.csv clf.pickle' % sys.argv[0])
    else:
        import pandas as pd
        from sklearn.cross_validation import train_test_split
        from nolearn.lasagne.visualize import plot_loss
        from math import sqrt
        from visualize import plot_images

        df = pd.read_csv(sys.argv[1])

        X = df.drop('label', axis=1).values.astype(np.float32) / 255.0
        y = df['label'].values.astype(np.int32)
        del df

        X_train, X_test, y_train, y_test = \
            train_test_split(X, y, test_size=0.2)

        imsize = (int(sqrt(X.shape[1])),) * 2
        del X, y

        X_train = X_train.reshape(-1, 1, imsize[0], imsize[1])
        X_test = X_test.reshape(-1, 1, imsize[0], imsize[1])

        clf = train(X_train, y_train)
        plot_loss(clf)

        y_pred = evaluate(clf, X_test, y_test)
        persist(clf, sys.argv[2])

        X_test = X_test.reshape(-1, imsize[0] * imsize[1])
        plot_images(X_test, y_test, y_pred)
