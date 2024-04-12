# Copyright (c) 2024 David Boetius
# Licensed under the MIT license
import pytest

from fairnessdatasets import Adult, AdultRaw


@pytest.mark.parametrize(
    "variable,expected_num_columns",
    [
        ("sex", 2),
        ("race", 5),
        ("relationship", 6),
        ("native-country", 41),
        ("age", 1),
    ],
)
def test_column_indices_adult(variable, expected_num_columns, adult_path):
    dataset = Adult(adult_path, train=False)
    assert len(dataset.column_indices(variable)) == expected_num_columns


@pytest.mark.parametrize(
    "variable",
    [
        "sex",
        "race",
        "relationship",
        "native-country",
        "age",
    ],
)
def test_sensitive_attribute_adult_raw(variable, adult_raw_path):
    dataset = AdultRaw(adult_raw_path, train=False)
    assert len(dataset.column_indices(variable)) == 1
