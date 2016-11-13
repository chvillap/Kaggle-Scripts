import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.ndimage import imread


def load_imgs(fnames, index):
    """
    """
    imgs = [imread(fn) for fn in fnames]
    return pd.Series(imgs, index=index)


def binarize_img(img, thresh=128):
    """
    """
    img[img < thresh] = 0
    img[img >= thresh] = 1
    return img


def pad_img(img, new_size):
    """
    """
    i0 = (new_size[0] - img.shape[0]) // 2
    j0 = (new_size[1] - img.shape[1]) // 2
    i1 = i0 + img.shape[0]
    j1 = j0 + img.shape[1]

    new_img = np.zeros(new_size, dtype=img.dtype)
    new_img[i0:i1, j0:j1] = img

    return new_img


def plot_mean_imgs(imgs, labels):
    """
    """
    from functools import reduce

    label_values = np.unique(labels)

    with PdfPages('mean_imgs.pdf') as pdf:
        for value in label_values:
            idx = (labels == value)

            max_w = reduce(max, [img.shape[1] for img in imgs[idx]])
            max_h = reduce(max, [img.shape[0] for img in imgs[idx]])

            mean_img = np.zeros((max_h, max_w), dtype=np.float32)
            for img in imgs[idx]:
                i1 = (max_h - img.shape[0]) // 2
                j1 = (max_w - img.shape[1]) // 2
                i2 = i1 + img.shape[0]
                j2 = j1 + img.shape[1]

                img_new = np.zeros((max_h, max_w), dtype=np.uint8)
                img_new[i1:i2, j1:j2] = img

                mean_img += img_new

            mean_img /= imgs.size

            plt.imshow(mean_img, cmap='gray')
            plt.title('Mean image: %s' % value)
            pdf.savefig()


if __name__ == '__main__':
    from os import listdir
    from os.path import join

    DATA_PATH = 'datasets'
    labels = pd.read_csv(join(DATA_PATH, 'train.csv'), index_col=0)
    labels = labels['species']

    IMG_PATH = join('datasets', 'images', 'train')
    fnames = sorted(listdir(IMG_PATH), key=lambda s: int(s.split('.')[0]))
    fnames = [join(IMG_PATH, fn) for fn in fnames]
    imgs = load_imgs(fnames, labels.index)
    imgs = imgs.apply(binarize_img)

    plot_mean_imgs(imgs, labels)
