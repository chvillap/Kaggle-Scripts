import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.externals import joblib

#______________________________________________________________________________

def train_classifier(dataset):
    """Trains a classifier using a randomized decision tree algorithm.
    Parameter tuning is performed using grid search and cross validation. The
    best combination of parameters found was n_estimators=55 and max_depth=8
    (although some randomness influence the results).
    """
    # clf = RandomForestClassifier(
    #     criterion='gini',
    #     max_features='sqrt',
    #     n_jobs=-1)

    # clf = ExtraTreesClassifier(
    #     criterion='gini',
    #     max_features='sqrt',
    #     n_jobs=-1)

    # clf = AdaBoostClassifier(
    #     base_estimator=DecisionTreeClassifier(
    #         criterion='gini',
    #         splitter='best'))

    clf = GradientBoostingClassifier(
        loss='deviance',
        max_features='sqrt',
        learning_rate=0.1)

    clf = GridSearchCV(
        estimator=clf,
        scoring='log_loss',
        # param_grid={ # RandomForestClassifier
        #     'n_estimators': [100, 200, 500],
        #     'max_depth': [2, 4, 6, 8, 10]},
        # param_grid={ # ExtraTreesClassifier
        #    'n_estimators': [100, 200, 500],
        #    'max_depth': [2, 4, 6, 8, 10]},
        # param_grid={ # AdaBoostClassifier + DecisionTreeClassifier
        #    'n_estimators': [100, 200, 500],
        #    'base_estimator__max_depth': [2, 4, 6, 8, 10],
        # param_grid={ # GradientBoostingClassifier
        #    'n_estimators': [60, 65, 70, 75, 80],
        #    'max_depth': [5, 6, 7, 8, 9]},
        param_grid={ # GradientBoostingClassifier
           'n_estimators': [50, 55, 60, 65, 70],
           'max_depth': [6, 7, 8, 9]},
        n_jobs=-1,
        verbose=2)

    y = dataset['OutcomeType']
    X = dataset.drop(['OutcomeType'], axis=1)
    clf.fit(X, y)

    for score in clf.grid_scores_:
        print(score)
    print('\nBest score:', clf.best_score_)
    print('Best parameters:', clf.best_params_)

    return clf


#______________________________________________________________________________

if __name__ == '__main__':
    from sys import argv

    if len(argv) < 3:
        print('Usage: {0} <dataset_in.csv> <clf_name>'.format(argv[0]))
    else:
        dataset = pd.read_csv(argv[1], index_col=0)

        clf = train_classifier(dataset)
        joblib.dump(clf, '{0}.pkl'.format(argv[2]))
