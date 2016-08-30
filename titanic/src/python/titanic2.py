import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.cross_validation import cross_val_score

#______________________________________________________________________________

def clean(dataset):
    """Cleans the dataset by transforming existing attributes and/or creating
    new attributes from them.
    """
    def get_title(name):
        n = len(titles)
        for i in range(n):
            if titles[i] in name:
                return i + 1
        return n + 1

    def get_stage_of_life(age):
        if age < 12.0:
            return 1 # child
        if age < 21.0:
            return 2 # young
        if age < 50.0:
            return 3 # adult
        return 4 # elder

    def get_cabin_prefix(cabin):
        if pd.isnull(cabin):
            return 0
        return cprefixes.index(cabin[0])


    # The "Ms." title has been introduced for women only in the 1970s, so no
    # one aboard Titanic could have been addressed as that. It might be just a
    # typo for "Mrs." or "Miss.". My guess is "Mrs.".
    dataset['Name'] = dataset['Name'].str.replace('Ms.', 'Mrs.')

    # Title ==> 1 (Master.), 2 (Miss.), 3 (Mr.), 4 (Mrs.), 5 (any other).
    titles = ['Master.', 'Miss.', 'Mr.', 'Mrs.']
    dataset['Title'] = dataset['Name'].apply(get_title)

    # Sex ==> -1 (female), 1 (male).
    dataset['Sex'] = dataset['Sex'].map({'female':-1, 'male':1}).astype(int)

    # Embarked ==> fill missing values with marginal mode.
    dataset.loc[dataset['Embarked'].isnull(), 'Embarked'] = \
        dataset['Embarked'].dropna().mode().values

    # Embarked ==> 1 (C), 2 (Q), 3 (S).
    dataset['Embarked'] = dataset['Embarked'].map({'C':1, 'Q':2, 'S':3}).astype(int)

    # # Age ==> fill missing values with median relative to Pclass, Sex and Title.
    # titles = np.unique(dataset['Title'].values)
    # pclasses = np.unique(dataset['Pclass'].values)
    # sexes = np.unique(dataset['Sex'].values)

    # med_ages = np.zeros((len(titles), len(pclasses), len(sexes)))
    # for i in range(len(titles)):
    #     for j in range(len(pclasses)):
    #         for k in range(len(sexes)):
    #             med_ages[i, j, k] = dataset.loc[
    #                 (dataset['Title'] == titles[i]) &
    #                 (dataset['Pclass'] == pclasses[j]) &
    #                 (dataset['Sex'] == sexes[k]), 'Age'].dropna().median()
    #             dataset.loc[
    #                 (dataset['Age'].isnull()) &
    #                 (dataset['Title'] == titles[i]) &
    #                 (dataset['Pclass'] == pclasses[j]) &
    #                 (dataset['Sex'] == sexes[k]), 'Age'] = med_ages[i, j, k]

    # Age ==> fill missing values with median relative to Title and Pclass.
    titles = np.unique(dataset['Title'].values)
    pclasses = np.unique(dataset['Pclass'].values)

    med_ages = np.zeros((titles.size, pclasses.size))
    for i in range(titles.size):
        for j in range(pclasses.size):
            med_ages[i, j] = dataset.loc[
                (dataset['Title'] == titles[i]) &
                (dataset['Pclass'] == pclasses[j]),
                'Age'].dropna().median()
            dataset.loc[
                (dataset['Age'].isnull()) &
                (dataset['Title'] == titles[i]) &
                (dataset['Pclass'] == pclasses[j]),
                'Age'] = med_ages[i, j]

    # Fare ==> fill missing values with median relative to Pclass.
    med_fares = np.zeros((pclasses.size,))
    for i in range(len(pclasses)):
        med_fares[i] = dataset.loc[
            dataset['Pclass'] == pclasses[i], 'Fare'].dropna().median()
        dataset.loc[(dataset['Fare'].isnull()) &
                    (dataset['Pclass'] == pclasses[i]), 'Fare'] = med_fares[i]

    # CabinDeck ==> 0 (missing), 1 (A), 2 (B), 3 (C), ..., 8 (T)
    cprefixes = ['', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'T']
    dataset['CabinDeck'] = dataset['Cabin'].apply(get_cabin_prefix)

    # Lifestage: 1 (child), 2 (young), 3 (adult), 4 (elder).
    dataset['Lifestage'] = dataset['Age'].apply(get_stage_of_life)

    # Relatives ==> combination (+) of SibSp and Parch.
    dataset['Relatives'] = dataset['SibSp'] + dataset['Parch']

    # Sex*Age ==> combination (*) of Sex and Age.
    dataset['Sex*Age'] = dataset['Sex'] * dataset['Age']

    # Sex*Age ==> combination (*) of Sex and Pclass.
    dataset['Sex*Pclass'] = dataset['Sex'] * dataset['Pclass']

    # Pclass*Fare ==> combination (*) of Sex and Lifestage.
    dataset['Sex*Lifestage'] = dataset['Sex'] * dataset['Lifestage']

    # Age*Pclass ==> combination (*) of Age and Pclass.
    dataset['Age*Pclass'] = dataset['Age'] * dataset['Pclass']

    # Pclass*Fare ==> combination (/) of Fare and Pclass.
    dataset['Fare/Pclass'] = dataset['Fare'] / dataset['Pclass']
    
    # # One-hot encoding.
    # # categorical = ['Sex', 'Embarked', 'Title', 'Lifestage']
    # categorical = ['Sex', 'Embarked', 'Title', 'Lifestage', 'CabinDeck']
    # for col in categorical:
    #     dummies = pd.get_dummies(dataset[col]).rename(
    #         columns=lambda x: '{0}_{1}'.format(col, x))
    #     dataset = dataset.join(dummies)

    # # CabinDeck has some value which appears in the training set but not in
    # # the test set, so the one hot encoding makes the cleaned test set have 1
    # # feature less than the cleaned training set. This fixes the problem.
    # data_cprefixes = pd.unique(dataset['CabinPrefix'])
    # for i in range(1, len(cprefixes)):
    #     if i not in data_cprefixes:
    #         dataset['CabinPrefix_{0}'.format(i)] = 0

    # Drop unused attributes.
    # unused = ['Name', 'Ticket', 'Cabin']
    unused = ['Name', 'Ticket', 'Cabin', 'Embarked', 'SibSp', 'Parch',
              'Lifestage', 'Pclass', 'Age']
    # unused = ['Name', 'Sex', 'Ticket', 'Cabin', 'SibSp', 'Parch', 'Title',
    #           'Embarked', 'Lifestage']
    # unused = ['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked', 'Title',
    #           'CabinPrefix', 'Lifestage']
    dataset = dataset.drop(unused, axis=1)

    # Save a copy.
    dataset.to_csv('cleaned.csv')

    return dataset


#______________________________________________________________________________

def train(dataset):
    """Trains a classifier model to fit the data. Parameter tuning is done
    using grid search and cross validation.
    """
    clf = RandomForestClassifier()

    # clf = GradientBoostingClassifier(
    #    loss='deviance')

    # clf = SVC(
    #     kernel='linear',
    #     decision_function_shape='ovr')

    clf = GridSearchCV(
        estimator=clf,
        scoring='log_loss',
        cv=10,
        iid=True,
        param_grid={
            'criterion': ['gini', 'entropy'],
            # 'learning_rate': [0.001, 0.01, 0.1],
            'n_estimators': [10, 50, 100, 150, 200],
            'max_depth': [2, 3, 4, 5, 6, 7, 8],
            'max_features': ['log2', 'sqrt', 0.2, 0.5, 0.7],
            'class_weight': [None, 'balanced']},
        # param_grid={
        #      'C': np.logspace(-2, 2, base=10, num=5),
        # #      'gamma': np.logspace(-5, 5, base=10, num=11)
        # },
        n_jobs=-1,
        verbose=2)

    X = dataset.drop(['Survived'], axis=1).values
    y = dataset['Survived'].values
    clf.fit(X, y)

    for score in clf.grid_scores_:
        print(score)
    print('\nBest score: {0}'.format(clf.best_score_))
    print('Best parameters: {0}'.format(clf.best_params_))

    clf.estimator.set_params(**clf.best_params_)
    clf.estimator.fit(X, y)

    feature_names = dataset.columns.drop('Survived')
    importances = list(zip(feature_names, clf.estimator.feature_importances_))
    importances.sort(key=lambda x: -x[1])

    print('\nFeature importances:')
    for name, value in importances:
        print('   {0} ==> {1}'.format(value, name))

    return clf


#______________________________________________________________________________

def predict(clf, dataset, y=None):
    """Makes predictions using a previously trained classifier.
    """
    X = dataset.values
    pred = clf.predict(X)

    if y != None:
        print('Classification report:\n', classification_report(y, pred))
        print('Confusion matrix:\n', confusion_matrix(y, pred))

    return pred


#______________________________________________________________________________

def write_submission_file(filename, dataset, pred):
    """Creates and writes the submission csv file.
    """
    out_df = pd.DataFrame(pred, columns=['Survived'], index=dataset.index)
    out_df.to_csv(filename)


#______________________________________________________________________________

if __name__ == '__main__':
    from sys import argv

    if len(argv) >= 3:
        train_dataset = pd.read_csv(argv[1], index_col=0)
        train_dataset = clean(train_dataset)
        clf = train(train_dataset)

        test_dataset = pd.read_csv(argv[2], index_col=0)
        test_dataset = clean(test_dataset)
        pred = predict(clf, test_dataset)

        write_submission_file("submission.csv", test_dataset, pred)
    else:
        print('Usage: {0} <train_dataset.csv> <test_dataset.csv>' \
            .format(argv[0]))
