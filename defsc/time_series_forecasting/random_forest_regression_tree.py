from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor


def generate_random_forest_regression_tree_model(train_x, train_y):
    regr = RandomForestRegressor(n_estimators=50, max_features=0.33)
    model = MultiOutputRegressor(regr).fit(train_x, train_y)
    return model
