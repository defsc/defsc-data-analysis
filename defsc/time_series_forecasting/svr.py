from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR


def generate_svr_regression_model(train_x, train_y):
    regr = SVR()
    model = MultiOutputRegressor(regr).fit(train_x, train_y)
    return model
