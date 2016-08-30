# =============================================================================
# visualization.py
# 
# Script to visualize data samples as images.
# =============================================================================


def view(df, ids, nrows, ncols, plot_keypts=True):
    """Plots the images and keypoints of multiple data samples.
    """
    def view_sub(row, ids_seq, plt_seq):
        """Plots the image and keypoints of a single data sample.
        """
        pixels = [int(x) for x in row[-1].split(' ')]
        image = np.reshape(pixels, (96, 96))

        plt.subplot(nrows, ncols, next(plt_seq))
        plt.title('Image %d' % next(ids_seq))
        plt.axis('off')
        plt.imshow(image, cmap='gray', interpolation='nearest')

        if plot_keypts:
            if row.size > 2:
                x = row[0:-1:2].values
                y = row[1:-1:2].values
                l = np.array([0, 0, 1, 2, 1, 2, 3, 4, 3, 4, 5, 6, 6, 7, 8])
                c = plt.cm.gist_rainbow(l / 8)
                plt.scatter(x=x, y=y, c=c, s=15, marker='x')
            else:
                print('Warning: No keypoints to plot')

    ids = ids[:nrows*ncols]
    df = df.ix[ids, :]

    ids_seq = iter(ids)
    plt_seq = iter(range(1, 1 + df.shape[0]))

    plt.figure()
    df.apply(view_sub, args=(ids_seq, plt_seq), axis=1)
    plt.show()


if __name__ == '__main__':
    from sys import argv

    if len(argv) < 5:
        print('Usage: %s dataset.csv nrows ncols id1 [id2 ...]' % argv[0])
    else:
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt

        df = pd.read_csv(argv[1])
        nrows = int(argv[2])
        ncols = int(argv[3])
        ids = [int(x) for x in argv[4:]]

        view(df, ids, nrows, ncols)
