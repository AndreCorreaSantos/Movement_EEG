To download the dataset simply run

```
wget -r -N -c -np https://physionet.org/files/eegmmidb/1.0.0/
```

On the project root.

Create env and install requirements.txt using

```
python -m venv 
```
run 
```
python preprocessing.py
```
to generate all preprocessed tf datasets.

Initial models taken from

https://github.com/ageron/handson-ml3/blob/main/15_processing_sequences_using_rnns_and_cnns.ipynb
