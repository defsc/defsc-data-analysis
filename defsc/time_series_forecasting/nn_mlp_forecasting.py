# https://machinelearningmastery.com/exploratory-configuration-multilayer-perceptron-network-time-series-forecasting/
from keras import Sequential
from keras.layers import Dense, Activation, Dropout

def generate_nn_mlp_model(train_x, train_y, test_x, test_y, number_of_hours_ahead, neurons=1, batch_size=4, epoch=50, verbose=2):
    model = Sequential()
    model.add(Dense(500, input_dim=train_x.shape[1]))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(Dense(250))
    model.add(Activation('relu'))
    model.add(Dense(number_of_hours_ahead))
    model.add(Activation('linear'))
    model.compile(optimizer='adam', loss='mse')

    model.fit(train_x, train_y, epochs=epoch, batch_size=batch_size, validation_data=(test_x, test_y), verbose=verbose,
              shuffle=False)

    return model
