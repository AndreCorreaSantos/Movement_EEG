import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM





def recurrent_model(features):
    model = Sequential()
    model.add(LSTM(units=64, activation="tanh", return_sequences=False, input_shape=(None, features)))
    model.add(Dense(units=3, activation="softmax"))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model




def wavenet(filters=32, kernel_size=2, dilation_rates=(1, 2, 4, 8) * 2):
    model = Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=[700, 64]))

    for rate in dilation_rates:
        model.add(tf.keras.layers.Conv1D(
            filters=filters, kernel_size=kernel_size, padding="causal", activation="relu",
            dilation_rate=rate))
        model.add(tf.keras.layers.Conv1D(filters=14, kernel_size=1))

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def convolutionalModel():
    model = Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=[700, 64]))
    model.add(tf.keras.layers.Conv1D(filters=32, kernel_size=2, padding="causal", activation="relu"))
    model.add(tf.keras.layers.Conv1D(filters=14, kernel_size=1))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

