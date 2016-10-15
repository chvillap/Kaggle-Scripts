# =============================================================================
# visualize.py
#
# Script with functions for plotting and visualizing the data.
# =============================================================================

import matplotlib.pyplot as plt

plt.style.use('bmh')


def plot_images(X, y=None, pred=None, gridsize=(5, 10)):
    """Plots all samples of the dataset as images, through a sequence of
    figures. If the labels and predictions are given, it also shows which
    predictions are correct and which are not.

    Inputs:
        X         Image data.
        y         Image labels.
        pred      Predicted labels.
        gridsize  Grid size (rows, columns) per figure.

    Outputs:
        (None)
    """
    from math import sqrt, ceil

    n_samples, n_features = X.shape
    n_samples_fig = gridsize[0] * gridsize[1]
    n_figures = ceil(n_samples / n_samples_fig)

    imsize = (int(sqrt(n_features)),) * 2

    for i in range(n_figures):
        print('Figure %d/%d' % (i + 1, n_figures))
        fig = plt.figure(i)

        for j in range(n_samples_fig):
            k = i * n_samples_fig + j
            
            if k >= X.shape[0]:
                break

            error = None
            color = 'black'
            if pred is not None and y is not None:
                error = y[k] != pred[k]
                color = 'red' if error else 'green'

            title = str(pred[k]) if pred is not None else ''

            plt.subplot(gridsize[0], gridsize[1], j + 1)
            plt.title(title, color=color)
            plt.axis('off')

            image = X[k].reshape(imsize)
            h = plt.imshow(image, interpolation='nearest', cmap='gray')

        fig.subplots_adjust(right=0.8)
        fig.colorbar(h, cax=fig.add_axes([0.85, 0.1, 0.02, 0.8]))

        plt.show()
