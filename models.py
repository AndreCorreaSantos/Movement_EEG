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

def convolutional():
    model = Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=[700, 64]))

    model.add(tf.keras.layers.Conv1D(filters=32, kernel_size=3, padding="causal", activation="relu"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling1D(2))

    model.add(tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding="causal", activation="relu"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling1D(2))

    model.add(tf.keras.layers.Conv1D(filters=128, kernel_size=3, padding="causal", activation="relu"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling1D(2))

    model.add(tf.keras.layers.Conv1D(filters=256, kernel_size=3, padding="causal", activation="relu"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling1D(2))

    model.add(tf.keras.layers.Conv1D(filters=256, kernel_size=3, padding="causal", activation="relu"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling1D(2))

    model.add(tf.keras.layers.Conv1D(filters=128, kernel_size=3, padding="causal", activation="relu"))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding="causal", activation="relu"))
    model.add(tf.keras.layers.BatchNormalization())

    # Flattening and final layers
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(100, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(3, activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

