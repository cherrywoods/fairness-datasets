# Copyright (c) 2023 David Boetius
# Licensed under the MIT license
import pytest

from adult import Adult, AdultRaw


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
def test_sensitive_attribute_adult(attributes, expected_num_columns, adult_path):
    dataset = Adult(adult_path, train=False, sensitive_attributes=attributes)
    assert len(dataset.sensitive_column_indices) == expected_num_columns


@pytest.mark.parametrize(
    "attributes,expected_num_columns",
    [
        ("sex", 1),
        ("race", 1),
        ("relationship", 1),
        ("native-country", 1),
        (("sex",), 1),
        (("sex", "race"), 2),
        (("sex", "race", "relationship"), 3),
        (("sex", "race", "relationship", "native-country"), 4),
    ],
)
def test_sensitive_attribute_adult_raw(
    attributes, expected_num_columns, adult_raw_path
):
    dataset = AdultRaw(adult_raw_path, train=False, sensitive_attributes=attributes)
    assert len(dataset.sensitive_column_indices) == expected_num_columns
