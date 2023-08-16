# Copyright (c) 2023 David Boetius
# Licensed under the MIT license
import pytest

from adult import Adult


@pytest.mark.parametrize(
    "training_set", [True, False], ids=["training set", "test set"]
)
def test_download_path(training_set: bool, tmp_path):
    dataset = Adult(root=tmp_path, train=training_set, download=True)
    assert len(dataset[0]) == 2


def test_download_string(tmp_path):
    dataset = Adult(root=str(tmp_path), train=False, download=True)
    assert len(dataset[0]) == 2
