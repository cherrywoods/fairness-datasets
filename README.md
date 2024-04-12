# fairness-datasets
PyTorch dataset wrappers for the several popular datasets from 
fair machine learning research.

The following datasets are wrapped:
 - [Adult (Census Income)](https://archive.ics.uci.edu/dataset/2/adult).
 - [Default](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)
 - [Law School](https://eric.ed.gov/?id=ED469370) (data from [here](https://www.tensorflow.org/responsible_ai/fairness_indicators/tutorials/Fairness_Indicators_Pandas_Case_Study))
 - [SouthGerman](https://archive.ics.uci.edu/dataset/573/south+german+credit+update)

## Installation
```shell
pip install adult-dataset
```

## Basic Usage
```python
from fairnessdatasets import Adult

# load (if necessary, download) the Adult training dataset 
train_set = Adult(root="datasets", download=True)
# load the test set
test_set = Adult(root="datasets", train=False, download=True)

inputs, target = train_set[0]  # retrieve the first sample of the training set

# iterate over the training set
for inputs, target in iter(train_set):
    ...  # Do something with a single sample

# use a PyTorch data loader
from torch.utils.data import DataLoader

loader = DataLoader(test_set, batch_size=32, shuffle=True)
for epoch in range(100):
    for inputs, targets in iter(loader):
        ...  # Do something with a batch of samples
```
You can use `Adult(..., raw=True)` to turn off the one-hot encoding
and z-score normalization applied by the `Adult` class by default.

The remaining dataset classes can be used in the same way as `Adult`.
However, these datasets don't come with a fixed train/test split, 
so that the dataset instances always contain all data.
To create a train/test split, use
```python
from fairnessdatasets import Default
from torch.utils.data import random_split

dataset = Default(root="datasets", download=True)

rng = torch.Generator().manual_seed(42)  # for reproducible results
train_set, test_set = random_split(dataset, [0.7, 0.3], generator=rng)
```

## Advanced Usage

Turn off status messages while downloading the dataset:
```python
Adult(root=..., output_fn=None)
```

Use the `logging` module for logging status messages while downloading the
dataset instead of placing the status messages on `sys.stdout`.
```python
import logging

Adult(root=..., output_fn=logging.info)
```
