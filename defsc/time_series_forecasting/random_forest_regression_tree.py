from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor


def generate_random_forest_regression_tree_model(train_x, train_y):
    regr = RandomForestRegressor()
    model = MultiOutputRegressor(regr).fit(train_x, train_y)
    return model
