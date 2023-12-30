# adult-dataset
A PyTorch dataset wrapper for the 
[Adult (Census Income)](https://archive.ics.uci.edu/dataset/2/adult) dataset.
Adult is a popular dataset in machine learning fairness research. 

This package provides the `adult.Adult` class:
a`torch.utils.data.Datasets` loading and, optionally, downloading the
Adult dataset.
It can be used like the `MNIST` dataset in
[torchvision](https://pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html?highlight=mnist#torchvision.datasets.MNIST).

Beyond `adult.Adult`, this package also provides `adult.AdultRaw`,
which works just as `adult.Adult`, but
does not standardize the features in the dataset and does not apply one-hot encoding.

## Installation
```shell
pip install adult-dataset
```

## Basic Usage
```python
from adult import Adult

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
