import numpy as np
from scipy.signal import spectrogram
from scipy.fft import fftshift
import matplotlib.pyplot as plt
import os
from concurrent.futures import ThreadPoolExecutor
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy




import mne

import numpy as np


def process_data(data,n_steps,individual,stride):

    def getXYdata(data):
        from sklearn.preprocessing import LabelEncoder
        yData = []
        X = data.get_data()
        contador = 0
        start_t = data.annotations.onset
        for i in range(0,X.shape[1]):
            time = i/160
            if contador < data.annotations.description.shape[0]:
                if time >= start_t[contador]:
                    contador = contador + 1
            yData.append(data.annotations.description[contador-1])
        yData = np.array(yData)
        le = LabelEncoder()
        le.fit(yData)
        yData = le.transform(yData)
        return X.T,yData

    Xdata,yData = getXYdata(data)
    features = 64  


    def splitTimeSeries(Xdata, ydata, n_steps,stride):
        X = np.zeros(((Xdata.shape[0] - n_steps)//stride, n_steps, features))
        y = np.zeros(((Xdata.shape[0] - n_steps)//stride, 1))


        for i in range(n_steps, len(X) - 1,stride):
            X[i - n_steps] = Xdata[i - n_steps:i]
            y[i - n_steps] = ydata[i+1]

        return X, y
    
    return splitTimeSeries(Xdata,yData,n_steps,stride)




def generator_function(basePath, individual, n_steps, stride):
    for exam in range(3, 15):  # valid tests for an individual are 3 to 14
        file = basePath + f"S{str(individual).zfill(3)}/S{str(individual).zfill(3)}R{str(exam).zfill(2)}.edf"
        data = mne.io.read_raw_edf(file)
        yield process_data(data, n_steps, individual, stride)

def create_tf_dataset(basePath, individual, n_steps, stride, batch_size):
    generator = lambda: generator_function(basePath, individual, n_steps, stride)
    dataset = tf.data.Dataset.from_generator(generator, output_signature=(
        tf.TensorSpec(shape=(1898,n_steps, 64), dtype=tf.float64),  # Remove None for batch size
        tf.TensorSpec(shape=(1898,1), dtype=tf.int64)  # Scalar label
    ))
    dataset = dataset.batch(batch_size)
    return dataset

# Example usage:
basePath = "physionet.org/files/eegmmidb/1.0.0/"
individual = 2
n_steps = 700
stride = 1
batch_size = 1

dataset = create_tf_dataset(basePath, individual, n_steps, stride, batch_size)

for i, ex in enumerate(dataset):
    print(i, ex[0].shape, ex[0].dtype)

# features = 64

# model = Sequential()
# model.add(LSTM(units=64, activation="tanh", return_sequences=False, input_shape=(n_steps, features)))  # Remove None for batch size
# model.add(Dense(units=3, activation="softmax"))

# model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# # More concise training loop using model.fit
# epochs = 3
# result = model.fit(dataset, epochs=epochs)

# plt.title("training results")
# plt.plot(result.history['loss'], label='train_loss')

# plt.plot(result.history['accuracy'], label='train_accuracy')
# plt.plot(result.history['val_accuracy'], label='val_accuracy')
# plt.legend()
# plt.show()  
