import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.externals import joblib
from sklearn.metrics import classification_report, confusion_matrix

#______________________________________________________________________________

def apply_classifier(clf, dataset):
    """Applies a previously trained classifier in the test data.
    """
    # X = dataset.drop(['OutcomeType'], axis=1)
    X = dataset
    pred = clf.predict(X)
    prob = clf.predict_proba(X)

    return pred, prob


#______________________________________________________________________________

def evaluate_classifier(clf, dataset, pred):
    """Evaluates the performance of the classifier in the test data.
    """
    X = dataset.drop(['OutcomeType'], axis=1)
    y = dataset['OutcomeType']

    print('Classification report:\n', classification_report(y, pred))
    print('Confusion matrix:\n', confusion_matrix(y, pred))


#______________________________________________________________________________

def write_submission_file(filename, dataset, prob):
    """Creates and writes the submission csv file.
    """
    columns = [
        'Adoption', 'Died', 'Euthanasia', 'Return_to_owner', 'Transfer']
    prob = pd.DataFrame(prob, columns=columns, index=dataset.index)
    prob.index.names = ['ID']
    prob.to_csv(filename, error_bad_lines=True)


#______________________________________________________________________________

if __name__ == '__main__':
    from sys import argv

    if len(argv) < 3:
        print('Usage: {0} <dataset_in.csv> <clf_name>'.format(argv[0]))
    else:
        dataset = pd.read_csv(argv[1], index_col=0)
        clf = joblib.load('{0}.pkl'.format(argv[2]))

        pred, prob = apply_classifier(clf, dataset)
        if 'OutcomeType' in dataset.columns:
            evaluate_classifier(clf, dataset, pred)
        
        write_submission_file('submission.csv', dataset, prob)
