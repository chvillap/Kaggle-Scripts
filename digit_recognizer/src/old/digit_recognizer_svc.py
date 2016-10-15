import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC, LinearSVC
from sklearn.grid_search import GridSearchCV
from sklearn.externals import joblib
from sklearn.metrics import classification_report, confusion_matrix

#______________________________________________________________________________

def read_dataset_csv(filename, targetname=None):
    """Reads dataset from csv file.
    Works for both labeled and unlabeled datasets. For labeled datasets, the
    name of the label (target) attribute should be passed as argument. 'None'
    means no target (unlabeled dataset). The data, target values and attribute
    names are returned in separate arrays.
    """
    print('Reading dataset "{0}"...'.format(filename))

    data = []
    target = []
    attrnames = []
    
    with open(filename, mode='r') as f:
        reader = csv.reader(f)

        attrnames = [s for s in next(reader)]
        t = attrnames.index(targetname) if targetname else -1

        for line in reader:
            if t >= 0:
                target.append(line[t])
                temp = line[:t] + line[t + 1:]
            else:
                temp = line
            data.append([int(s) for s in temp])

    return np.array(data), np.array(target), np.array(attrnames)



#______________________________________________________________________________

def write_submission_csv(filename, fieldnames, predict):
    """Writes submission csv file with the results.
    """
    print('Writing submission "{0}"...'.format(filename))

    with open(filename, 'w') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_NONE)
        writer.writerow(fieldnames)

        numsamples = predict.shape[0]
        for i in range(numsamples):
            writer.writerow((i + 1, predict[i]))


#______________________________________________________________________________

def plot_dataset(data, target=np.array([]), predict=np.array([]), rw=4, cl=7):
    """Plots a series of samples as NxN images, as well as the predictions made
    for each of them if this information is available. When both the predicted
    and expected target values are given, predicted values are shown in
    different colors (red if incorrect, blue otherwise). If there are too many
    samples to be seen, their visualization is divided into multiple figures
    ("pages"). The number of samples per figure can be manually specified using
    parameters rw and cl.
    """
    print('Plotting samples...')
    
    from math import sqrt, ceil

    numsamples, numattr = data.shape
    samples_per_page = rw * cl
    numpages = ceil(numsamples / samples_per_page)

    for p in range(numpages):
        print('Page {0}/{1}'.format(p + 1, numpages))
        plt.figure(p)

        for k in range(samples_per_page):
            i = p * samples_per_page + k
            imgsize = int(sqrt(numattr))
            image = np.reshape(data[i], (imgsize, imgsize))

            error = target.size and predict.size and target[i] != predict[i]
            title = str(predict[i]) if predict.size else ''
            titlecolor = 'red' if error else 'blue'

            plt.subplot(rw, cl, k + 1)
            plt.imshow(image, interpolation='nearest', cmap='gray')
            plt.title(title, color=titlecolor)
            plt.axis('off')

        plt.show()


#______________________________________________________________________________

def train_clf(data, target, name='clf'):
    """Trains a Support Vector Classifier (SVC) to fit the training data.
    Parameter tuning is done using grid search and 3-fold cross-validation, but
    you can skip that by just setting C=100 and gamma=4.9E-7, which was the best
    combination of parameters (for now).
    After the training, the resulting classifier is written into a set of files
    that can be easily read and used for predictions later.
    """
    print('Training the classifier...')
    
    # clf = GridSearchCV(
    #     estimator=SVC(
    #         kernel='rbf',
    #         C=100,
    #         decision_function_shape='ovr'),
    #     param_grid={
    #         'gamma': [1E-7, 2.5E-7, 5E-7, 7.5E-7, 1E-6]},
    #     n_jobs=-1,
    #     verbose=2)

    # clf = GridSearchCV(
    #     estimator=LinearSVC(class_weight='balanced'),
    #     param_grid={
    #         'C': np.logspace(-9, -4, base=10, num=6)},
    #     n_jobs=-1,
    #     verbose=2)

    clf = SVC(kernel='rbf',
              C=100,
              gamma=4.9E-7,
              decision_function_shape='ovr')

    # clf = LinearSVC(
    #         C=100,
    #         verbose=2)

    clf.fit(data, target)
    joblib.dump(clf, '{0}.pkl'.format(name))

    # for grid_score in clf.grid_scores_:
    #     print(grid_score)
    # print('\nBest score:', clf.best_score_)
    # print('Best parameters:', clf.best_params_)

    return clf


#______________________________________________________________________________

def test_clf(data, target=np.array([]), name='clf'):
    """Tests the Support Vector Classifier (SVC) over the test data.
    The classifier must have been previously trained and saved. The test data
    can be either labeled or unlabeled. When it is labeled, performance metrics
    are employed to evaluate the classifier. These metrics are the mean
    accuracy (score), confusion matrix, precision, recall, and the f1-score.
    """
    print('Testing the classifier...')

    clf = joblib.load('{0}.pkl'.format(name))
    predict = clf.predict(data)

    results = {}
    if target.size:
        results['score'] = clf.score(data, target)
        results['confusion_matrix'] = confusion_matrix(target, predict)
        results['report'] = classification_report(target, predict)

        print('Score:', results['score'])
        print('Confusion matrix:\n', results['confusion_matrix'])
        print('Other metrics:\n', results['report'])

    return predict, results


#______________________________________________________________________________

if __name__ == '__main__':
    import sys

    if len(sys.argv) >= 3:
        command = sys.argv[1].upper()
        if command not in ['TRAIN', 'TEST']:
            print('Invalid command. Should be one of the following:')
            print('  TRAIN - Fits a model to the training data')
            print('  TEST  - Predicts outcomes for the test data')
            sys.exit(1)

        if command == 'TRAIN':
            data, target, attrnames = \
                read_dataset_csv(sys.argv[2], targetname='label')
            model = train_clf(data=data, target=target, name='svc_clf')

        elif command == 'TEST':
            data, target, attrnames = read_dataset_csv(sys.argv[2])
            predict, results = test_clf(data=data, target=target, name='svc_clf')

            write_submission_csv('submission.csv', ('ImageId', 'Label'), predict)
            plot_dataset(data, target, predict)
    else:
        print('Usage:', sys.argv[0], '<command> <dataset.csv>')
