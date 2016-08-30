import numpy as np
import pandas as pd
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
    def find_title(name):
        for i in range(len(titles)):
            if titles[i] in name:
                return i
        return len(titles)

    def set_stage_of_life(age):
        if age < 12.0:
            return 0 # child
        if age < 21.0:
            return 1 # young
        return 2 # adult


    # The "Ms." title has been introduced for women only in the 1970s, so no
    # one aboard Titanic could have been addressed as that. It must be just a
    # typo for "Mrs." or "Miss.". My guess is "Mrs.".
    dataset['Name'] = dataset['Name'].str.replace('Ms.', 'Mrs.')

    # Title ==> 0 (Master.), 1 (Miss.), 2 (Mr.), 3 (Mrs.), 4 (any other).
    titles = ['Master.', 'Miss.', 'Mr.', 'Mrs.']
    dataset['Title'] = dataset['Name'].apply(find_title)

    # Sex ==> -1 (female), 1 (male).
    dataset['Sex'] = dataset['Sex'].map({'female':0, 'male':1}).astype(int)

    # Embarked ==> fill missing values with marginal mode.
    dataset.loc[dataset['Embarked'].isnull(), 'Embarked'] = \
        dataset['Embarked'].dropna().mode().values

    # Embarked ==> 0 (C), 1 (Q), 2 (S).
    dataset['Embarked'] = dataset['Embarked'].map({'C':0, 'Q':1, 'S':2}).astype(int)

    # Age ==> fill missing values with median relative to Pclass, Sex and Title.
    titles = np.unique(dataset['Title'].values)
    pclasses = np.unique(dataset['Pclass'].values)
    sexes = np.unique(dataset['Sex'].values)

    med_ages = np.zeros((len(titles), len(pclasses), len(sexes)))
    for i in range(len(titles)):
        for j in range(len(pclasses)):
            for k in range(len(sexes)):
                med_ages[i, j, k] = dataset.loc[
                    (dataset['Title'] == titles[i]) &
                    (dataset['Pclass'] == pclasses[j]) &
                    (dataset['Sex'] == sexes[k]), 'Age'].dropna().median()
                dataset.loc[
                    (dataset['Age'].isnull()) &
                    (dataset['Title'] == titles[i]) &
                    (dataset['Pclass'] == pclasses[j]) &
                    (dataset['Sex'] == sexes[k]), 'Age'] = med_ages[i, j, k]

    # Fare ==> fill missing values with median relative to Pclass.
    med_fares = np.zeros((len(pclasses),))
    for i in range(len(pclasses)):
        med_fares[i] = dataset.loc[
            dataset['Pclass'] == pclasses[i], 'Fare'].dropna().median()
        dataset.loc[(dataset['Fare'].isnull()) &
                    (dataset['Pclass'] == pclasses[i]), 'Fare'] = med_fares[i]

    # Relatives ==> combination (+) of SibSp and Parch.
    dataset['Relatives'] = dataset['SibSp'] + dataset['Parch']

    # Age*Pclass ==> combination (*) of Age and Pclass.
    dataset['Age*Pclass'] = dataset['Age'] * dataset['Pclass']

    # StageOfLife: 0 (child), 1 (young), 2 (adult).
    dataset['StageOfLife'] = dataset['Age'].apply(set_stage_of_life)
    
    # One-hot encoding.
    categorical = ['Sex', 'Title', 'Embarked', 'StageOfLife']
    for col in categorical:
        dummies = pd.get_dummies(dataset[col]).rename(
            columns=lambda x: '{0}_{1}'.format(col, x))
        dataset = dataset.join(dummies)

    # Drop unused attributes.
    unused = ['Name', 'Sex', 'Ticket', 'Cabin', 'SibSp', 'Parch', 'Title',
              'Embarked', 'StageOfLife']
    dataset = dataset.drop(unused, axis=1)

    # Save a copy.
    dataset.to_csv('cleaned.csv')

    return dataset


#______________________________________________________________________________

def train(dataset):
    """Trains a classifier model to fit the data. Parameter tuning is done
    using grid search and cross validation.
    """
    clf = RandomForestClassifier(
        criterion='gini')

    # clf = GradientBoostingClassifier(
    #     loss='deviance',
    #     learning_rate=0.1)

    # clf = SVC(
    #     kernel='linear',
    #     decision_function_shape='ovr')

    clf = GridSearchCV(
        estimator=clf,
        scoring='accuracy',
        param_grid={
           'n_estimators': [50, 75, 100, 125, 150, 175, 200],
           'max_depth': [2, 3, 4, 5, 6, 7, 8, 9, None],
           'max_features': [0.2, 0.4, 0.6, 0.8]},
        # param_grid={
        #     'C': np.logspace(-2, 2, base=10, num=5),
        #     #'gamma': np.logspace(-5, 5, base=10, num=11)
        # },
        n_jobs=-1,
        verbose=2)

    X = dataset.drop(['Survived'], axis=1).values
    y = dataset['Survived'].values
    clf.fit(X, y)

    for score in clf.grid_scores_:
        print(score)
    print('\nBest score:', clf.best_score_)
    print('Best parameters:', clf.best_params_)

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
