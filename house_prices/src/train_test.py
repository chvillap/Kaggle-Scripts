# =============================================================================
# train_test.py
#
# Script to train and test a predictive model using preprocessed data.
# =============================================================================

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import make_scorer

np.random.seed(42)


def rmsle(y, y_pred):
    """Computes the root mean squared logarithmic error (RMSLE) metric.
    """
    msle = np.sum((np.log1p(y) - np.log1p(y_pred))**2) / np.prod(y_pred.shape)
    return np.sqrt(msle)


def plot_learning_curves(estimators, title, X, y, ylim=None, cv=None,
                         scoring=make_scorer(rmsle, greater_is_better=False),
                         train_sizes=np.linspace(0.1, 1, 5), n_jobs=-1):
    """Plots training and validation errors for increasingly bigger portions of
    the data set, showing the learning progress.
    """
    from sklearn.model_selection import learning_curve

    palette = sns.color_palette()

    for i in range(len(estimators)):
        train_sizes, train_scores, valid_scores = learning_curve(
            estimators[i], X, y, cv=cv, scoring=scoring,
            train_sizes=train_sizes, n_jobs=n_jobs)

        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        valid_scores_mean = np.mean(valid_scores, axis=1)
        valid_scores_std = np.std(valid_scores, axis=1)

        sns.plt.plot(train_sizes, train_scores_mean, 'o--', color=palette[i],
            label=type(estimators[i]).__name__ + ' (train)')
        sns.plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
            train_scores_mean + train_scores_std, alpha=0.1, color=palette[i])

        sns.plt.plot(train_sizes, valid_scores_mean, 'o-', color=palette[i],
            label=type(estimators[i]).__name__ + ' (cv)')
        sns.plt.fill_between(train_sizes, valid_scores_mean - valid_scores_std,
            valid_scores_mean + valid_scores_std, alpha=0.1, color=palette[i])

    sns.plt.grid('on')
    sns.plt.legend(loc='best')
    sns.plt.show()


def select_features(X, y, feature_names=None):
    """Uses a LASSO regression model to select the most relevant features.
    """
    from sklearn.feature_selection import SelectFromModel
    from sklearn.linear_model import LassoCV

    lasso_cv = LassoCV(alphas=np.logspace(-5, 5, 21), random_state=42)
    selector = SelectFromModel(lasso_cv)
    X_sel = selector.fit_transform(X, y)

    if feature_names:
        selected = np.array(feature_names)[selector.get_support()].tolist()
        print('Selected features: %d', len(selected))
        print(selected)

    return X_sel, selector


def train_lasso(X, y):
    """Trains a tuned LASSO regression model.
    """
    from sklearn.linear_model import Lasso
    from sklearn.model_selection import GridSearchCV

    lasso_gs = GridSearchCV(
        estimator=Lasso(max_iter=10000, random_state=42),
        scoring=make_scorer(rmsle, greater_is_better=False),
        param_grid={
            'alpha': np.logspace(-5, 5, 21)
        },
        cv=10,
        n_jobs=-1,
        verbose=1,
    ).fit(X, y)

    print('Best score:', lasso_gs.best_score_)
    print('Best params:', lasso_gs.best_params_)

    return lasso_gs.best_estimator_.fit(X, y)


def train_xgboost(X, y):
    """Trains a tuned XGBoost regression model.
    """
    from xgboost import XGBRegressor
    from sklearn.model_selection import GridSearchCV

    # # Tune n_estimators.
    # xgb_gs = GridSearchCV(
    #     estimator=XGBRegressor(
    #         max_depth=4,
    #         min_child_weight=1,
    #         gamma=0,
    #         subsample=0.8,
    #         colsample_bytree=0.8,
    #         seed=42),
    #     scoring=make_scorer(rmsle, greater_is_better=False),
    #     param_grid={
    #         'n_estimators': np.linspace(100, 1000, 10, dtype=int)
    #     },
    #     cv=10,
    #     n_jobs=-1,
    #     verbose=1,
    # ).fit(X, y)
    # print('Best score:', xgb_gs.best_score_)
    # print('Best params:', xgb_gs.best_params_)

    # # Tune max_depth and min_child_weight.
    # xgb_gs = GridSearchCV(
    #     estimator=XGBRegressor(
    #         gamma=0,
    #         subsample=0.8,
    #         colsample_bytree=0.8,
    #         seed=42),
    #     scoring=make_scorer(rmsle, greater_is_better=False),
    #     param_grid={
    #         'n_estimators': [xgb_gs.best_params_['n_estimators']],
    #         'max_depth': np.linspace(2, 6, 5, dtype=int),
    #         'min_child_weight': np.linspace(1, 3, 5),
    #     },
    #     cv=10,
    #     n_jobs=-1,
    #     verbose=1,
    # ).fit(X, y)
    # print('Best score:', xgb_gs.best_score_)
    # print('Best params:', xgb_gs.best_params_)

    # # Tune gamma.
    # xgb_gs = GridSearchCV(
    #     estimator=XGBRegressor(
    #         subsample=0.8,
    #         colsample_bytree=0.8,
    #         seed=42),
    #     scoring=make_scorer(rmsle, greater_is_better=False),
    #     param_grid={
    #         'n_estimators': [xgb_gs.best_params_['n_estimators']],
    #         'max_depth': [xgb_gs.best_params_['max_depth']],
    #         'min_child_weight': [xgb_gs.best_params_['min_child_weight']],
    #         'gamma': np.linspace(0, 0.1, 11),
    #     },
    #     cv=10,
    #     n_jobs=-1,
    #     verbose=1,
    # ).fit(X, y)
    # print('Best score:', xgb_gs.best_score_)
    # print('Best params:', xgb_gs.best_params_)

    # # Tune subsample and colsample_bytree.
    # xgb_gs = GridSearchCV(
    #     estimator=XGBRegressor(
    #         seed=42),
    #     scoring=make_scorer(rmsle, greater_is_better=False),
    #     param_grid={
    #         'n_estimators': [xgb_gs.best_params_['n_estimators']],
    #         'max_depth': [xgb_gs.best_params_['max_depth']],
    #         'min_child_weight': [xgb_gs.best_params_['min_child_weight']],
    #         'gamma': [xgb_gs.best_params_['gamma']],
    #         'subsample': np.linspace(0.6, 0.9, 4),
    #         'colsample_bytree': np.linspace(0.6, 0.9, 4),
    #     },
    #     cv=10,
    #     n_jobs=-1,
    #     verbose=1,
    # ).fit(X, y)
    # print('Best score:', xgb_gs.best_score_)
    # print('Best params:', xgb_gs.best_params_)

    # # Tune reg_alpha and reg_lambda.
    # xgb_gs = GridSearchCV(
    #     estimator=XGBRegressor(
    #         seed=42),
    #     scoring=make_scorer(rmsle, greater_is_better=False),
    #     param_grid={
    #         'n_estimators': [xgb_gs.best_params_['n_estimators']],
    #         'max_depth': [xgb_gs.best_params_['max_depth']],
    #         'min_child_weight': [xgb_gs.best_params_['min_child_weight']],
    #         'gamma': [xgb_gs.best_params_['gamma']],
    #         'subsample': [xgb_gs.best_params_['subsample']],
    #         'colsample_bytree': [xgb_gs.best_params_['colsample_bytree']],
    #         'reg_alpha': np.linspace(0, 1, 4),
    #         'reg_lambda': np.linspace(0, 1, 4),
    #     },
    #     cv=10,
    #     n_jobs=-1,
    #     verbose=1,
    # ).fit(X, y)
    # print('Best score:', xgb_gs.best_score_)
    # print('Best params:', xgb_gs.best_params_)

    # # Set a lower learning_rate and increase n_estimators.
    # xgb = XGBRegressor(
    #     learning_rate=0.01,
    #     n_estimators=xgb_gs.best_params_['n_estimators'] * 10,
    #     max_depth=xgb_gs.best_params_['max_depth'],
    #     min_child_weight=xgb_gs.best_params_['min_child_weight'],
    #     gamma=xgb_gs.best_params_['gamma'],
    #     subsample=xgb_gs.best_params_['subsample'],
    #     colsample_bytree=xgb_gs.best_params_['colsample_bytree'],
    #     reg_alpha=xgb_gs.best_params_['reg_alpha'],
    #     reg_lambda=xgb_gs.best_params_['reg_lambda'],
    #     seed=42,
    # )

    # Final estimator (best score).
    xgb = XGBRegressor(
        learning_rate=0.005,
        n_estimators=10000,
        max_depth=3,
        min_child_weight=1,
        gamma=0,
        subsample=0.9,
        colsample_bytree=0.6,
        reg_alpha=0,
        reg_lambda=1,
        seed=42,
    )

    return xgb.fit(X, y)


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 4:
        print('Usage: %s train.csv test.csv submission.csv' % sys.argv[0])
    else:
        df_train = pd.read_csv(sys.argv[1], index_col=0, na_values=[''],
                               keep_default_na=False)
        df_test = pd.read_csv(sys.argv[2], index_col=0, na_values=[''],
                              keep_default_na=False)

        X_train = df_train.drop('SalePrice__log1p', axis=1).values
        y_train = df_train['SalePrice__log1p'].values
        X_test = df_test.values

        feature_names = df_train.columns.tolist()
        # X_train_sel, selector = select_features(X_train, y_train, feature_names)
        # X_test_sel = selector.transform(X_test)
        X_train_sel = X_train
        X_test_sel = X_test

        lasso = train_lasso(X_train_sel, y_train)
        xgb = train_xgboost(X_train_sel, y_train)

        plot_learning_curves([lasso, xgb], 'Learning Curves',
                             X_train_sel, y_train, cv=5, n_jobs=-1)

        y_test_lasso = np.expm1(lasso.predict(X_test_sel))
        y_test_xgb = np.expm1(xgb.predict(X_test_sel))
        y_test = 0.65 * y_test_lasso + 0.35 * y_test_xgb

        df_submission = pd.DataFrame(
            data={'SalePrice': y_test,
                  'SalePrice_lasso': y_test_lasso,
                  'SalePrice_xgb': y_test_xgb},
            index=df_test.index,
        )
        df_submission[['SalePrice']].to_csv(sys.argv[3])

        print(df_submission.head(10))
