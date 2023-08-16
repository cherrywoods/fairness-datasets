# Copyright (c) 2023 David Boetius
# Licensed under the MIT license
import pytest
from tempfile import TemporaryDirectory

from adult import Adult


@pytest.fixture(scope="module")
def dataset_path():
    with TemporaryDirectory() as tmp_dir:
        Adult(tmp_dir, train=False, download=True)
        yield tmp_dir


@pytest.mark.parametrize(
    "attributes,expected_num_columns",
    [
        ("sex", 2),
        ("race", 5),
        ("relationship", 6),
        ("native-country", 41),
        (("sex",), 2),
        (("sex", "race"), 7),
        (("sex", "race", "relationship"), 13),
    ],
)
def test_sensitive_attribute(attributes, expected_num_columns, dataset_path):
    dataset = Adult(dataset_path, train=False, sensitive_attributes=attributes)
    assert len(dataset.sensitive_column_indices) == expected_num_columns
