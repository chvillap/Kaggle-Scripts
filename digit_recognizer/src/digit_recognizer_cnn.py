import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sknn.mlp import Classifier, Convolution, Layer
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.externals import joblib

empty = np.empty(shape=(0,))


#______________________________________________________________________________

def plot_imgs(X, y=empty, pred=empty, gsize=(5, 10)):
    """Plots all samples of the dataset as images through a series of figures.
    If the labels (y) and predictions (pred) are available, it also shows which
    predictions were correct and which were not.
    """
    from math import sqrt, ceil

    n_samples, n_features = X.shape
    n_samples_fig = gsize[0] * gsize[1]
    n_figures = ceil(n_samples / n_samples_fig)

    imgsize = (int(sqrt(n_features)),) * 2

    for i in range(n_figures):
        print('Figure {0}/{1}'.format(i + 1, n_figures))
        fig = plt.figure(i)

        for j in range(n_samples_fig):
            k = i * n_samples_fig + j
            
            error = y.size and pred.size and y[k] != pred[k]
            title = str(pred[k]) if pred.size else ''
            color = 'red' if error else 'blue'

            plt.subplot(gsize[0], gsize[1], j + 1)
            plt.title(title, color=color)
            plt.axis('off')
            
            img = X[k].reshape(imgsize)
            ih = plt.imshow(img, interpolation='nearest', cmap='gray')

        fig.subplots_adjust(right=0.8)
        fig.colorbar(ih, cax=fig.add_axes([0.85, 0.1, 0.02, 0.8]))

        plt.show()


#______________________________________________________________________________

def select_features(X, y):
    """Select a subset of features according to the 80th percentile of the
    highest scores in the univariate chi-square test.
    """
    sel = SelectPercentile(chi2, percentile=80)
    X_sel = sel.fit_transform(X, y)

    return X_sel, sel


#______________________________________________________________________________

def train_clf(X, y):
    """Trains the classifier using a multilayer neural network model.
    Optimal parameters are determined through grid search and 3-fold cross
    validation.
    """
    clf = Classifier(
        layers=[
            Convolution(
                type='Rectifier',
                kernel_shape=(3, 3),
                channels=8,
                pool_type='max',
                pool_shape=(2, 2)),
            Convolution(
                type='Rectifier',
                kernel_shape=(3, 3),
                channels=8),
            Layer(
                type='Sigmoid',
                units=500),
            Layer(
                type='Softmax')],
        regularize='L2',
        learning_rate=0.002,
        batch_size=10,
        valid_size=0.2,
        verbose=True)

    clf.fit(X, y)

    # for grid_score in clf.grid_scores_:
    #     print(grid_score)
    # print('\nBest score: {0}'.format(clf.best_score_))
    # print('Best parameters: {0}'.format(clf.best_params_))

    return clf


#______________________________________________________________________________

def test_clf(clf, X, y=empty):
    """Tests the k-nearest neighbors classifier and evaluate its performance
    using different metrics.
    """
    pred = clf.predict(X)

    if y != empty:
        y = y.reshape(pred.shape)
        print('Accuracy: {0}'.format((y == pred).sum() / y.size))
        print('Confusion matrix:\n', confusion_matrix(y, pred))
        print('Classification report:\n', classification_report(y, pred))

    return pred


#______________________________________________________________________________

def write_submission_file(index, results):
    """Writes the submission csv file.
    """
    df = pd.DataFrame(results, index=index, columns=['Label'])
    df.index.names = ['ImageId']
    df.to_csv('submission.csv')


#______________________________________________________________________________

if __name__ == '__main__':
    from sys import argv

    if len(argv) < 3:
        print('Usage: {0} <train-or-test> <dataset.csv> <classifier>'.format(argv[0]))
    else:
        command = argv[1].lower()

        if command == 'train':
            df = pd.read_csv(argv[2])

            train_size = int(0.9 * df.shape[0])
            X_train = df.drop(['label'], axis=1).head(train_size).values
            y_train = df['label'].head(train_size).values
            # X_train = df.drop(['label'], axis=1).values
            # y_train = df['label'].values
            
            test_size = df.shape[0] - train_size
            X_test = df.drop(['label'], axis=1).tail(test_size).values
            y_test = df['label'].tail(test_size).values

            X_train_sel = X_train
            # X_train_sel, sel = select_features(X_train, y_train)
            clf = train_clf(X_train_sel, y_train)

            X_test_sel = X_test
            # X_test_sel = sel.transform(X_test)
            test_clf(clf, X_test_sel, y_test)

            # joblib.dump(sel, 'sel.pkl')
            joblib.dump(clf, '{0}.pkl'.format(argv[3]))

        elif command == 'test':
            df = pd.read_csv(argv[2])

            if 'label' in df.columns:
                X_test = df.drop(['label'], axis=1).values
                y_test = df['label'].values
            else:
                X_test = df.values
                y_test = empty

            # sel = joblib.load('sel.pkl')
            clf = joblib.load('{0}.pkl'.format(argv[3]))

            X_test_sel = X_test
            # X_test_sel = sel.transform(X_test)
            pred = test_clf(clf, X_test_sel, y_test)

            write_submission_file(df.index + 1, pred)
            plot_imgs(X_test, y_test, pred)

        else:
            raise ValueError('Invalid command (should be "train" or "test")')
