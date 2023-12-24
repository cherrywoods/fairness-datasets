# Copyright (c) 2023 David Boetius
# Licensed under the MIT license
import pytest

from adult import Adult, AdultRaw


@pytest.mark.parametrize(
    "training_set", [True, False], ids=["training set", "test set"]
)
@pytest.mark.parametrize("adult_class", [Adult, AdultRaw], ids=["Adult", "AdultRaw"])
def test_download_path(training_set: bool, adult_class, tmp_path):
    dataset = adult_class(root=tmp_path, train=training_set, download=True)
    assert len(dataset[0]) == 2


def test_download_string(tmp_path):
    dataset = Adult(root=str(tmp_path), train=False, download=True)
    assert len(dataset[0]) == 2
