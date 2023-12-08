

import mne
import numpy as np
import tensorflow as tf



def getXYdata(data):
    from sklearn.preprocessing import LabelEncoder
    yData = []
    X = data.get_data()
    count = 0
    start_t = data.annotations.onset
    for i in range(0,X.shape[1]):
        time = i/160
        if count < data.annotations.description.shape[0]:
            if time >= start_t[count]:
                count = count + 1
        yData.append(data.annotations.description[count-1])
    yData = np.array(yData)
    le = LabelEncoder()
    le.fit(yData)
    yData = le.transform(yData)
    return X.T,yData




def splitTimeSeries(Xdata, ydata, n_steps,stride):

    for i in range(n_steps, len(Xdata) - 1,stride):
        X = Xdata[i - n_steps:i]
        y = ydata[i+1]

        yield X, y
    

def generator_function(basePath,n_steps, stride):
    for individual in range(1, 110):  # valid individuals are 1 to 109
        for exam in range(3, 15):  # valid tests for an individual are 3 to 14
            file = basePath + f"S{str(individual).zfill(3)}/S{str(individual).zfill(3)}R{str(exam).zfill(2)}.edf"
            data = mne.io.read_raw_edf(file)
            Xdata, yData = getXYdata(data)
            yield from splitTimeSeries(Xdata, yData, n_steps, stride)

def create_tf_dataset(basePath, n_steps, stride, batch_size):
    generator = lambda: generator_function(basePath, n_steps, stride)
    dataset = tf.data.Dataset.from_generator(generator, output_signature=(
        tf.TensorSpec(shape=(n_steps, 64), dtype=tf.float64),
        tf.TensorSpec(shape=(), dtype=tf.int64)
    ))
    dataset = dataset.batch(batch_size)
    return dataset



if __name__ == "__main__":
    basePath = "physionet.org/files/eegmmidb/1.0.0/"
    n_steps = 700
    stride = 100
    batch_size = 1

    full_dataset = create_tf_dataset(basePath, n_steps, stride, batch_size)
    # take
    for X, y in full_dataset.take(1):
        print(X.shape)
        print(y.shape)
    # saving dataset 
    # tf.data.experimental.save(full_dataset, 'dataset')