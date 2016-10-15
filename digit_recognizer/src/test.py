# =============================================================================
# test.py
#
# Script with functions for testing the classifier on unseen data.
# =============================================================================

import numpy as np
import pandas as pd
import sys


def load(name):
    """Loads a pretrained classifier from files.

    Inputs:
        name  Classifier name.

    Outputs:
        clf  Classifier object
    """
    import pickle

    sys.setrecursionlimit(10000)
    with open('%s.pickle' % name, 'rb') as f:
        clf = pickle.load(f)

    return clf


def write_submission(name, y):
    """Writes the submission file using the prediction results.

    Inputs:
        name  Submission file name.
        y     Predicted labels.

    Outputs:
        (None)
    """
    index = pd.Series(np.arange(1, y.size + 1), name='ImageId')
    df = pd.DataFrame({'Label': y}, index=index)
    df.to_csv('%s.csv' % name)


if __name__ == '__main__':
    if len(sys.argv) < 4:
        print('Usage: %s dataset.csv clf.pickle submission.csv' % sys.argv[0])
    else:
        from visualize import plot_images
        from math import sqrt

        X = pd.read_csv(sys.argv[1]).values.astype(np.float32) / 255.0

        imsize = (int(sqrt(X.shape[1])),) * 2
        X = X.reshape(-1, 1, imsize[0], imsize[1])

        clf = load(sys.argv[2])
        y_pred = clf.predict(X)

        write_submission(sys.argv[3], y_pred)

        X = X.reshape(-1, imsize[0] * imsize[1])
        plot_images(X, pred=y_pred)
