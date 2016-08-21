# =============================================================================
# test.py
#
# Script for testing some pretrained model on unseen data.
# =============================================================================


def get_shaped_data(df, clf_id):
    """Puts data in shape for the classifiers.
    """
    X = df.values.astype(np.float32)
    X = X.reshape(-1, 1, 96, 96)
    y = np.empty((X.shape[0], 0), dtype=np.float32)
    
    return X, y


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print('Usage: %s dataset.csv' % sys.argv[0])
    else:
        import numpy as np
        import pandas as pd
        import pickle

        df = pd.read_csv(sys.argv[1], index_col=0)
        X, y = get_shaped_data(df)
        del df

        sys.setrecursionlimit(10000)
        for clf_id in range(1, 10):
            with open('models/net_%d.pickle' % clf_id, 'rb') as f:
                model = pickle.load(f)
                y = np.hstack((y, model.predict(X)))

        c1 = [6, 8, 7, 9, 14, 16, 15, 17]
        c2 = [8, 6, 9, 7, 16, 14, 17, 15]
        y[:, c1] = y[:, c2]

        df = pd.DataFrame(y.reshape((y.size, 1)), columns=['Location'])
        df.index.name = 'RowId'
        df.to_csv('submission.csv')
