# Copyright (c) 2024 David Boetius
# Licensed under the MIT license
import pytest
from torch.utils.data import DataLoader

from fairnessdatasets import Adult, SouthGerman, Default


@pytest.fixture(
    scope="module",
    params=[
        (Adult, {"raw": False}, "adult_path"),
        (Adult, {"raw": True}, "adult_raw_path"),
        (SouthGerman, {"raw": False}, "south_german_path"),
        (SouthGerman, {"raw": True}, "south_german_raw_path"),
        (Default, {"raw": False}, "default_path"),
        (Default, {"raw": True}, "default_raw_path"),
    ],
    ids=[
        "Adult",
        "Adult-raw",
        "SouthGerman",
        "SouthGerman-raw",
        "Default",
        "Default-raw",
    ],
)
def dataset(request):
    dataset_class, init_kwargs, dataset_path = request.param
    dataset_path = request.getfixturevalue(dataset_path)
    return dataset_class(dataset_path, **init_kwargs)


def test_iterate(dataset):
    for i, (inputs, target) in enumerate(iter(dataset)):
        if i % 100 == 0:
            print(i, inputs, target)
        if i > 500:
            break


def test_data_loader(dataset):
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    for i, (inputs, targets) in enumerate(iter(data_loader)):
        if i % 100 == 0:
            print(i, inputs, targets)
