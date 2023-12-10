import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional, Conv1D, MaxPooling1D, Flatten, InputLayer
from tensorflow.keras.regularizers import l2


def recurrent(features,n_steps=200):
    model = Sequential()
    model.add(LSTM(units=64, activation="tanh", return_sequences=False, input_shape=(None, features)))
    model.add(Dense(units=3, activation="softmax"))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def recurrent_2(features, lstm_units=64, dropout_rate=0.5):
    model = Sequential()

    model.add(Bidirectional(LSTM(units=lstm_units, activation="tanh", return_sequences=True), input_shape=(None, features)))

    model.add(BatchNormalization())

    model.add(LSTM(units=lstm_units, activation="tanh", return_sequences=False))

    model.add(Dropout(dropout_rate))

    model.add(Dense(units=3, activation="softmax"))

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def wavenet(filters=32, kernel_size=2, dilation_rates=(1, 2, 4, 8) * 2,n_steps=200):
    model = Sequential()
    # Adjusted input shape to (sequence_length, num_features)
    model.add(InputLayer(input_shape=(n_steps, 64)))

    for rate in dilation_rates:
        model.add(Conv1D(
            filters=filters, kernel_size=kernel_size, padding="causal", activation="relu",
            dilation_rate=rate))
        model.add(Conv1D(filters=14, kernel_size=1))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    # Removed the extra input_shape parameter
    model.add(Dense(3, activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def convolutional(features, n_steps=200):
    model = Sequential()
    model.add(InputLayer(input_shape=(n_steps, features)))

    model.add(Conv1D(filters=32, kernel_size=3, padding="causal", activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(2))

    model.add(Conv1D(filters=64, kernel_size=3, padding="causal", activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(2))

    model.add(Conv1D(filters=128, kernel_size=3, padding="causal", activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(2))

    model.add(Conv1D(filters=256, kernel_size=3, padding="causal", activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(2))

    model.add(Conv1D(filters=256, kernel_size=3, padding="causal", activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(2))

    model.add(Conv1D(filters=128, kernel_size=3, padding="causal", activation="relu"))
    model.add(BatchNormalization())

    model.add(Conv1D(filters=64, kernel_size=3, padding="causal", activation="relu"))
    model.add(BatchNormalization())

    # Flattening and final layers
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(3, activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def convolutional_2(features, n_steps=200, dropout_rate=0.5, l2_penalty=0.001):
    model = Sequential()
    model.add(InputLayer(input_shape=(n_steps, features)))

    model.add(Conv1D(filters=32, kernel_size=3, padding="causal", activation="relu", kernel_regularizer=l2(l2_penalty)))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(2))
    model.add(Dropout(dropout_rate))

    model.add(Conv1D(filters=64, kernel_size=3, padding="causal", activation="relu", kernel_regularizer=l2(l2_penalty)))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(2))
    model.add(Dropout(dropout_rate))

    model.add(Conv1D(filters=128, kernel_size=3, padding="causal", activation="relu", kernel_regularizer=l2(l2_penalty)))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(2))
    model.add(Dropout(dropout_rate))

    model.add(Conv1D(filters=256, kernel_size=3, padding="causal", activation="relu", kernel_regularizer=l2(l2_penalty)))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(2))
    model.add(Dropout(dropout_rate))

    model.add(Conv1D(filters=256, kernel_size=3, padding="causal", activation="relu", kernel_regularizer=l2(l2_penalty)))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(2))
    model.add(Dropout(dropout_rate))

    model.add(Conv1D(filters=128, kernel_size=3, padding="causal", activation="relu", kernel_regularizer=l2(l2_penalty)))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))

    model.add(Conv1D(filters=64, kernel_size=3, padding="causal", activation="relu", kernel_regularizer=l2(l2_penalty)))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))

    # Flattening and final layers
    model.add(Flatten())
    model.add(Dense(100, activation='relu', kernel_regularizer=l2(l2_penalty)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(3, activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
