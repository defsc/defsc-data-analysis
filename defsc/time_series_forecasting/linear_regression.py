from sklearn.multioutput import MultiOutputRegressor
from sklearn import linear_model


def generate_linear_regression_model(train_x, train_y):
    regr = linear_model.LinearRegression()
    model = MultiOutputRegressor(regr).fit(train_x, train_y)
    return model
