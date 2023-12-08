

import mne
import numpy as np
import tensorflow as tf
import argparse



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



def parse_args():
    parser = argparse.ArgumentParser(description="Create TF dataset with specified parameters.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for the dataset.")
    parser.add_argument("--stride", type=int, default=100, help="Stride value for splitting time series.")
    parser.add_argument("--n_steps", type=int, default=700, help="Number of steps for each instance.")
    return parser.parse_args()

if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_args()

    # Assign command-line arguments to variables
    batch_size = args.batch_size
    stride = args.stride
    n_steps = args.n_steps

    basePath = "physionet.org/files/eegmmidb/1.0.0/"

    full_dataset = create_tf_dataset(basePath, n_steps, stride, batch_size)

    # Determine the number of elements in 80% of the dataset
    dataset_size = tf.data.experimental.cardinality(full_dataset).numpy()
    train_size = int(0.8 * dataset_size)

    # Create the training dataset
    train_dataset = full_dataset.take(train_size)
    tf.data.experimental.save(train_dataset, 'train_dataset')

    # Create the testing dataset
    test_dataset = full_dataset.skip(train_size)
    tf.data.experimental.save(test_dataset, 'test_dataset')

    # Create smaller datasets (20% of original size)
    tiny_train_size = int(0.2 * train_size)
    tiny_test_size = int(0.2 * (dataset_size - train_size))

    # Take 20% of the training dataset as the tiny training dataset
    tiny_train_dataset = train_dataset.take(tiny_train_size)
    tf.data.experimental.save(tiny_train_dataset, 'tiny_train_dataset')

    # Take 20% of the testing dataset as the tiny testing dataset
    tiny_test_dataset = test_dataset.take(tiny_test_size)
    tf.data.experimental.save(tiny_test_dataset, 'tiny_test_dataset')

    # Print shapes for verification
    for X, y in tiny_train_dataset.take(1):
        print("Tiny Training Dataset - X shape:", X.shape)
        print("Tiny Training Dataset - y shape:", y.shape)

    for X, y in tiny_test_dataset.take(1):
        print("Tiny Testing Dataset - X shape:", X.shape)
        print("Tiny Testing Dataset - y shape:", y.shape)