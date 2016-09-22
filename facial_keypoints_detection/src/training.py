# =============================================================================
# training.py
#
# Script to train convolutional neural networks for regression.
# =============================================================================


CLF_SETTINGS = {
    1: {
        'targets': ['left_eye_center_x',
                    'left_eye_center_y',
                    'right_eye_center_x',
                    'right_eye_center_y'],
    },
    2: {
        'targets': ['left_eye_inner_corner_x',
                    'left_eye_inner_corner_y',
                    'right_eye_inner_corner_x',
                    'right_eye_inner_corner_y'],
    },
    3: {
        'targets': ['left_eye_outer_corner_x',
                    'left_eye_outer_corner_y',
                    'right_eye_outer_corner_x',
                    'right_eye_outer_corner_y'],
    },
    4: {
        'targets': ['left_eyebrow_inner_end_x',
                    'left_eyebrow_inner_end_y',
                    'right_eyebrow_inner_end_x',
                    'right_eyebrow_inner_end_y'],
    },
    5: {
        'targets': ['left_eyebrow_outer_end_x',
                    'left_eyebrow_outer_end_y',
                    'right_eyebrow_outer_end_x',
                    'right_eyebrow_outer_end_y'],
    },
    6: {
        'targets': ['nose_tip_x',
                    'nose_tip_y'],
    },
    7: {
        'targets': ['mouth_left_corner_x',
                    'mouth_left_corner_y',
                    'mouth_right_corner_x',
                    'mouth_right_corner_y'],
    },
    8: {
        'targets': ['mouth_center_top_lip_x',
                    'mouth_center_top_lip_y'],
    },
    9: {
        'targets': ['mouth_center_bottom_lip_x',
                    'mouth_center_bottom_lip_y'],
    },
}


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
    """Stops the learning process early if the model spends too many epochs
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


def get_shaped_data(df, clf_id):
    """Puts data in shape for some specialized classifier.
    """
    data_cols = [('pixel_%d' % i) for i in range(96 ** 2)]
    target_cols = CLF_SETTINGS[clf_id]['targets']

    df = df.dropna(subset=target_cols)

    X = df[data_cols].values.astype(np.float32)
    X = X.reshape(-1, 1, 96, 96)
    y = df[target_cols].values.astype(np.float32)
    
    return X, y


def train(X, y):
    """Trains the neural network model using some training data.
    """
    from theano import shared
    from nolearn.lasagne import NeuralNet
    from nolearn.lasagne import TrainSplit
    from lasagne.layers import InputLayer
    from lasagne.layers import Conv2DLayer
    from lasagne.layers import DenseLayer
    from lasagne.layers import MaxPool2DLayer
    from lasagne.layers import DropoutLayer

    layers = [
        (InputLayer, {'shape': (None, X.shape[1], X.shape[2], X.shape[3])}),
        (Conv2DLayer, {'num_filters': 32, 'filter_size': 3, pad='same'}),
        (Conv2DLayer, {'num_filters': 32, 'filter_size': 3, pad='same'}),
        (MaxPool2DLayer, {'pool_size': 2}),
        (Conv2DLayer, {'num_filters': 64, 'filter_size': 3, pad='same'}),
        (Conv2DLayer, {'num_filters': 64, 'filter_size': 3, pad='same'}),
        (MaxPool2DLayer, {'pool_size': 2}),
        (Conv2DLayer, {'num_filters': 128, 'filter_size': 5, pad='same'}),
        (Conv2DLayer, {'num_filters': 128, 'filter_size': 3, pad='same'}),
        (MaxPool2DLayer, {'pool_size': 2}),
        (DropoutLayer, {'p': 0.5}),
        (DenseLayer, {'num_units': 512}),
        (DropoutLayer, {'p': 0.5}),
        (DenseLayer, {'num_units': 512}),
        (DenseLayer, {'num_units': y.shape[1], 'nonlinearity': None}),
    ]
    model = NeuralNet(
        layers,
        # update=<function nesterov_momentum at 0x7f14b1fe9b70>,
        # loss=None,
        # objective=<function objective at 0x7f14b1f78510>,
        # objective_loss_function=None,
        # batch_iterator_train=<nolearn.lasagne.base.BatchIterator object at 0x7f14b1f745f8>,
        # batch_iterator_test=<nolearn.lasagne.base.BatchIterator object at 0x7f14b1f74668>,
        regression=True,
        max_epochs=10,
        #max_epochs=1000,
        train_split=TrainSplit(eval_size=0.25),
        # custom_score=None,
        # X_tensor_type=None,
        # y_tensor_type=None,
        # use_label_encoder=False,
        on_epoch_finished=[
            AdjustParameter('update_learning_rate', start=0.01, stop=0.00001),
            AdjustParameter('update_momentum', start=0.9, stop=0.999),
            EarlyStopping(patience=100),
        ],
        # on_training_started=None,
        # on_training_finished=None,
        verbose=2,
        update_learning_rate=shared(0.01),
        update_momentum=shared(0.9),
    )

    return model.fit(X, y)


def evaluate(model, X_test, y_test):
    """Evaluates a pretrained model using some test data.
    """
    from sklearn.metrics import mean_absolute_error
    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import explained_variance_score
    from sklearn.metrics import r2_score

    y_pred = model.predict(X_test)

    # r2 = 1 - sum((y_test - y_pred)**2) / sum((y_test - y_test.mean(axis=0))**2)
    # ev = 1 - (y_test - y_pred).var(axis=0) / y_test.var(axis=0)
    # mae = sum(abs(y_test - y_pred)) / y_test.shape[0]
    # mse = sum((y_test - y_pred)**2) / y_test.shape[0]
    # rmse = np.sqrt(mse)

    r2 = r2_score(y_test, y_pred, multioutput='raw_values')
    ev = explained_variance_score(y_test, y_pred, multioutput='raw_values')
    mae = mean_absolute_error(y_test, y_pred, multioutput='raw_values')
    mse = mean_squared_error(y_test, y_pred, multioutput='raw_values')
    rmse = np.sqrt(mse)
    
    print('+-----------------------------------------------+')
    print('  RESULTS:')
    print('    R^2 score:                      {}'.format(r2))
    print('    Explained variance score:       {}'.format(ev))
    print('    Mean absolute error (MAE):      {}'.format(mae))
    print('    Mean squared error (MSE):       {}'.format(mse))
    print('    Root mean squared error (RMSE): {}'.format(rmse))
    print('+-----------------------------------------------+')


def plot_learning_curves(model, clf_id=1):
    """Plots the learning curves of a pretrained model.
    """
    import matplotlib.pyplot as plt

    colors = [
        '#f8766d', '#d39200', '#93aa00',
        '#00ba38', '#00c19f', '#00b9e3',
        '#619cff', '#db72fb', '#ff61c3']

    train_loss = np.array([x['train_loss'] for x in model.train_history_])
    valid_loss = np.array([x['valid_loss'] for x in model.train_history_])
    train_loss = np.sqrt(train_loss)
    valid_loss = np.sqrt(valid_loss)

    fig = plt.figure()
    plt.plot(train_loss, label='Training set', color=colors[clf_id-1],
        linewidth=2, linestyle='--')
    plt.plot(valid_loss, label='Validation set', color=colors[clf_id-1],
        linewidth=2, linestyle='-')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.yscale('log')
    plt.grid('on')
    plt.legend()
    plt.title('Learning curves (net_%d)' % clf_id)
    fig.savefig('net_%d.pdf' % clf_id)


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 3:
        print('Usage: %s dataset.csv clf_id' % sys.argv[0])
        print('  +--------+--------------------------------------+')
        print('  | clf_id |           TARGET VARIABLES           |')
        print('  +--------+--------------------------------------+')
        print('  |    1   | [left,right]_eye_center_[x,y]        |')
        print('  |    2   | [left,right]_eye_inner_corner_[x,y]  |')
        print('  |    3   | [left,right]_eye_outer_corner_[x,y]  |')
        print('  |    4   | [left,right]_eyebrow_inner_end_[x,y] |')
        print('  |    5   | [left,right]_eyebrow_outer_end_[x,y] |')
        print('  |    6   | nose_tip_[x,y]                       |')
        print('  |    7   | mouth_[left,right]_corner_[x,y]      |')
        print('  |    8   | mouth_center_top_lip_[x,y]           |')
        print('  |    9   | mouth_center_bottom_lip_[x,y]        |')
        print('  +--------+--------------------------------------+')
    else:
        import numpy as np
        import pandas as pd
        import pickle
        from sklearn.cross_validation import train_test_split

        df = pd.read_csv(sys.argv[1])
        clf_id = int(sys.argv[2])

        X, y = get_shaped_data(df, clf_id)
        del df

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        model = train(X_train, y_train)
        evaluate(model, X_test, y_test)
        plot_learning_curves(model, clf_id)

        sys.setrecursionlimit(10000)
        with open('models/net_%d.pickle' % clf_id, 'wb') as f:
            pickle.dump(model, f, -1)
