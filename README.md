To download the dataset simply run

```
wget -r -N -c -np https://physionet.org/files/eegmmidb/1.0.0/
```

On the project root.

Create env and install requirements.txt using

```
python -m venv 
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

Initial analysis and models taken from

https://github.com/ageron/handson-ml3/blob/main/15_processing_sequences_using_rnns_and_cnns.ipynb


