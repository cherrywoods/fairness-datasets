# Copyright (c) 2023 David Boetius
# Licensed under the MIT license
from tempfile import TemporaryDirectory

import pytest
from torch.utils.data import DataLoader

from adult import Adult, AdultRaw


@pytest.fixture(
    scope="module",
    params=[(Adult, "adult_path"), (AdultRaw, "adult_raw_path")],
    ids=["Adult", "AdultRaw"],
)
def dataset(request):
    adult_class, dataset_path = request.param
    dataset_path = request.getfixturevalue(dataset_path)
    return adult_class(dataset_path)


def test_iterate(dataset):
    for i, (inputs, target) in enumerate(iter(dataset)):
        if i % 1000 == 0:
            print(i, inputs, target)


def test_data_loader(dataset):
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    for i, (inputs, targets) in enumerate(iter(data_loader)):
        if i % 100 == 0:
            print(i, inputs, targets)
