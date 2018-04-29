# https://machinelearningmastery.com/exploratory-configuration-multilayer-perceptron-network-time-series-forecasting/
from keras import Sequential
from keras.layers import Dense, Activation, Dropout

def generate_nn_mlp_model(train_x, train_y, test_x, test_y, number_of_hours_ahead, neurons=4, batch_size=2, epoch=100, verbose=2):

    model = Sequential()

    model.add(Dense(int(train_x.shape[1]/2), input_dim=train_x.shape[1], activation='relu'))
    model.add(Dense(number_of_hours_ahead))
    model.compile(loss='mae', optimizer='adam')

    #model.fit(train_x, train_y, epochs=epoch, batch_size=batch_size, validation_data=(test_x, test_y), verbose=verbose,
    model.fit(train_x, train_y, epochs=epoch, batch_size=batch_size, verbose=verbose,
              shuffle=False)

    return model
