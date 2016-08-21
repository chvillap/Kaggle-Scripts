# =============================================================================
# preprocessing.py
#
# Script to transform and prepare data for the learning algorithm.
# =============================================================================


if __name__ == '__main__':
    from sys import argv

    if len(argv) < 3:
        print('Usage: %s dataset.csv out.csv' % argv[0])
    else:
        import pandas as pd
        import numpy as np
        
        df = pd.read_csv(argv[1])

        npixels = 96**2
        oldcolumns = df.columns[:-1].tolist()
        newcolumns = [('pixel_%d' % i) for i in range(npixels)]

        # # Faster, but requires more memory.
        # data = df['Image'].str.split(expand=True)
        # data = data.astype(np.float32) / 255
        # df.drop(['Image'], axis=1, inplace=True)
        # df2 = pd.DataFrame(data, index=df.index)
        # df2 = pd.concat([df, df2], axis=1)
        # df2.columns = oldcolumns + newcolumns
        # df2.to_csv(argv[2], index=False)

        # Much slower, but unavoidable if you have low memory available.
        nrows = df.shape[0]
        ncols = df.shape[1] + npixels - 1
        df2 = pd.DataFrame(np.empty((nrows, ncols), dtype=int),
            index=df.index, columns=oldcolumns + newcolumns)
        for i in range(nrows):
            df2.ix[i, newcolumns] = \
                [np.float32(x) / 255 for x in df.ix[i, 'Image'].split()]
        df2.ix[:, oldcolumns] = df.ix[:, oldcolumns]
        df2.to_csv(argv[2], index=False)
