# EEGMMIDB Dataset and Preprocessing

## Overview

This repository provides instructions for downloading and preprocessing the EEGMMIDB dataset, a dataset commonly used for EEG (Electroencephalogram) analysis. The dataset is hosted on PhysioNet and contains EEG recordings from subjects performing motor imagery and executed movements.

## Getting Started

### Downloading the Dataset

To download the dataset, use the following command in the project root:

```bash
wget -r -N -c -np https://physionet.org/files/eegmmidb/1.0.0/
```

This command uses `wget` to recursively download the necessary files from the PhysioNet repository.

### Setting up the Environment

Create a virtual environment and install the required packages by running the following commands:

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
pip install -r requirements.txt
```

This ensures that you have a clean environment with all the dependencies.

### Preprocessing the Dataset

Run the preprocessing script to generate preprocessed TensorFlow datasets:

```bash
python preprocessing.py
```
Create the tf dataset running
```
python preprocessing.py
```
It is possible to change the values passed to each variable, but any changes will have to be adjusted in the model loader in the master.ipynb.
Example: 
```
python preprocessing.py --batch_size 2 --stride 100 --n_steps 700
```

This script handles the necessary steps to preprocess the EEGMMIDB dataset and prepares it for further analysis.

## References

The initial models used in this project are sourced from the following repository:

[https://github.com/ageron/handson-ml3/blob/main/15_processing_sequences_using_rnns_and_cnns.ipynb](https://github.com/ageron/handson-ml3/blob/main/15_processing_sequences_using_rnns_and_cnns.ipynb)

These models serve as a starting point for further development and experimentation with the EEGMMIDB dataset.

https://github.com/ageron/handson-ml3/blob/main/15_processing_sequences_using_rnns_and_cnns.ipynb
