# Copyright (c) 2023 David Boetius
# Licensed under the MIT license
from tempfile import TemporaryDirectory

import pytest
from torch.utils.data import DataLoader

from adult import Adult


@pytest.fixture(scope="module")
def dataset_path():
    with TemporaryDirectory() as tmp_dir:
        Adult(tmp_dir, train=True, download=True)
        yield tmp_dir


def test_iterate(dataset_path):
    dataset = Adult(dataset_path)
    for i, (inputs, target) in enumerate(iter(dataset)):
        if i % 1000 == 0:
            print(i, inputs, target)


def test_data_loader(dataset_path):
    dataset = Adult(dataset_path)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    for i, (inputs, targets) in enumerate(iter(data_loader)):
        if i % 100 == 0:
            print(i, inputs, targets)
