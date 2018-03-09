from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM


def reshape_input_for_lstm(x, number_of_hours_backward, number_of_x_vars):
    reshaped_x = x.reshape((x.shape[0], number_of_hours_backward, number_of_x_vars))

    return reshaped_x


def generate_nn_lstm_model(train_x, train_y, test_x, test_y, number_of_hours_ahead):
    model = Sequential()
    model.add(LSTM(50, input_shape=(train_x.shape[1], train_x.shape[2])))
    model.add(Dense(number_of_hours_ahead))
    model.compile(loss='mae', optimizer='adam')

    history = model.fit(train_x, train_y, epochs=50, batch_size=72, validation_data=(test_x, test_y), verbose=2,
                      shuffle=False)

    return model
