

import mne
import numpy as np
import tensorflow as tf
import argparse



def getXYdata(data): 
    """ Extracts X and y data from mne.io.read_raw_edf object and encodes the labels"""
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
    """ Splits time series into n_steps instances with stride"""

    for i in range(n_steps, len(Xdata) - 1,stride):
        X = Xdata[i - n_steps:i]
        y = ydata[i+1]

        yield X, y
    

def generator_function(basePath,n_steps, stride):
    """ Generator function that yields X and y data from the files in the basePath directory"""
    for individual in range(1, 110):  # valid individuals are 1 to 109
        for exam in range(3, 15):  # valid tests for an individual are 3 to 14
            file = basePath + f"S{str(individual).zfill(3)}/S{str(individual).zfill(3)}R{str(exam).zfill(2)}.edf"
            data = mne.io.read_raw_edf(file)
            Xdata, yData = getXYdata(data)
            yield from splitTimeSeries(Xdata, yData, n_steps, stride)

def create_tf_dataset(basePath, n_steps, stride, batch_size):
    """ Creates a tensorflow dataset from the generator function"""
    generator = lambda: generator_function(basePath, n_steps, stride)
    dataset = tf.data.Dataset.from_generator(generator, output_signature=(
        tf.TensorSpec(shape=(n_steps, 64), dtype=tf.float64),
        tf.TensorSpec(shape=(), dtype=tf.int64)
    ))
    dataset = dataset.batch(batch_size)
    return dataset



def parse_args():
    parser = argparse.ArgumentParser(description="Create TF dataset with specified parameters.")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for the dataset.")
    parser.add_argument("--stride", type=int, default=100, help="Stride value for splitting time series.")
    parser.add_argument("--n_steps", type=int, default=200, help="Number of steps for each instance.")
    parser.add_argument("--tiny",type=float,default=0.05,help="Tiny dataset size")
    parser.add_argument("--save",type=bool,default=False,help="Save full train and test datasets")
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

    dataset_size = full_dataset.reduce(0, lambda x,_: x+1).numpy()
    train_size = int(0.8 * dataset_size)


    train_dataset = full_dataset.take(train_size)
    test_dataset = full_dataset.skip(train_size)
    if args.save:
        tf.data.experimental.save(train_dataset, 'train_dataset')
        tf.data.experimental.save(test_dataset, 'test_dataset')

    # Create smaller datasets (20% of original size)
    tiny_train_size = int(args.tiny * train_size)
    tiny_test_size = int(args.tiny * (dataset_size - train_size))

    if not args.save:
        tiny_train_dataset = train_dataset.take(tiny_train_size)
        tiny_test_dataset = test_dataset.take(tiny_test_size)
        tf.data.experimental.save(tiny_test_dataset, 'tiny_test_dataset')
        tf.data.experimental.save(tiny_train_dataset, 'tiny_train_dataset')