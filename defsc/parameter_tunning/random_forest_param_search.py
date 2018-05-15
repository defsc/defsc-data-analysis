from datetime import time

import numpy as np
from pandas import to_datetime, read_csv
from sklearn.ensemble import RandomForestRegressor
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline

from defsc.data_structures_transformation.data_structures_transformation import split_timeseries_set_on_test_train, \
    transform_dataframe_to_supervised
from defsc.filtering.time_series_cleaning import simple_fill_missing_values
from defsc.parameter_tunning import report


def multioutput_random_forest_regression_params_search(train_x, train_y):
    pipe = Pipeline([('reg', MultiOutputRegressor(RandomForestRegressor()))])

    param_grid = {'reg__estimator__n_estimators': [int(x) for x in np.linspace(start=200, stop=2000, num=100)],
                  'reg__estimator__max_features': [1, 3, 10, 'auto', 'sqrt'],
                  'reg__estimator__max_depth': [3, None],
                  'reg__estimator__min_samples_split': [2, 3, 10],
                  'reg__estimator__min_samples_leaf': [1, 3, 10],
                  'reg__estimator__bootstrap': [True, False]}

    grid_search(pipe, param_grid, train_x, train_y)

    random_grid = {'reg__estimator__n_estimators': [int(x) for x in np.linspace(start=200, stop=2000, num=100)],
                   'reg__estimator__max_features': ['auto', 'sqrt'],
                   'reg__estimator__max_depth': [int(x) for x in np.linspace(10, 110, num=11)] + [None],
                   'reg__estimator__min_samples_split': [2, 5, 10],
                   'reg__estimator__min_samples_leaf': [1, 2, 4],
                   'reg__estimator__bootstrap': [True, False]}

    randomized_search(pipe, random_grid, train_x, train_y)


def grid_search(estimator, parameters, train_x, train_y):
    grid_search = GridSearchCV(estimator, param_grid=parameters)
    start = time()
    grid_search.fit(train_x, train_y)

    print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
          % (time() - start, len(grid_search.cv_results_['params'])))

    report(grid_search.cv_results)


def randomized_search(estimator, parameters, train_x, train_y, n_iter=100, cv=3, verbose=2, random_state=42, n_jobs=4):
    random_search = RandomizedSearchCV(estimator=estimator,
                                       param_distributions=parameters,
                                       n_iter=n_iter,
                                       cv=cv,
                                       verbose=verbose,
                                       random_state=random_state,
                                       n_jobs=n_jobs)

    start = time()
    random_search.fit(train_x, train_y)
    print("RandomizedSearchCV took %.2f seconds for %d candidates"
          " parameter settings." % ((time() - start), n_iter))
    report(random_search.cv_results)


def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                results['mean_test_score'][candidate],
                results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")


if __name__ == "__main__":
    csv = '../data/multivariate-time-series/raw-210.csv'
    df = read_csv(csv, header=0, index_col=0)
    df.index = to_datetime(df.index)

    df = simple_fill_missing_values(df)

    number_of_timestep_ahead = 24
    number_of_timestep_backward = 24

    x_column_names = df.columns
    y_column_names = ['airly-pm1']

    df = transform_dataframe_to_supervised(df, x_column_names, y_column_names, number_of_timestep_ahead,
                                           number_of_timestep_backward)

    df = df.dropna()

    train_x, train_y, test_x, test_y = split_timeseries_set_on_test_train(df.values,
                                                                          len(
                                                                              x_column_names) * number_of_timestep_backward,
                                                                          len(
                                                                              y_column_names) * number_of_timestep_ahead)

    multioutput_random_forest_regression_params_search(train_x, train_y)
