# Copyright (c) 2024 David Boetius
# Licensed under the MIT license
import pytest

from fairnessdatasets import Adult, Default, LawSchool, SouthGerman


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
    dataset = Adult(adult_raw_path, train=False, raw=True)
    assert len(dataset.column_indices(variable)) == 1


@pytest.mark.parametrize(
    "dataset_class,init_kwargs,dataset_path,variable,expected_num_columns",
    [
        (Adult, {}, "adult_path", "sex", 2),
        (Adult, {}, "adult_path", "race", 5),
        (Adult, {}, "adult_path", "relationship", 6),
        (Adult, {}, "adult_path", "native-country", 41),
        (Adult, {}, "adult_path", "age", 1),
        (Adult, {}, "adult_path", "hours-per-week", 1),
        (Adult, {"raw": True}, "adult_raw_path", "sex", 1),
        (Adult, {"raw": True}, "adult_raw_path", "race", 1),
        (Adult, {"raw": True}, "adult_raw_path", "relationship", 1),
        (Adult, {"raw": True}, "adult_raw_path", "native-country", 1),
        (Adult, {"raw": True}, "adult_raw_path", "age", 1),
        (Adult, {"raw": True}, "adult_raw_path", "hours-per-week", 1),
        (Default, {}, "default_path", "SEX", 2),
        (Default, {}, "default_path", "EDUCATION", 4),
        (Default, {}, "default_path", "MARRIAGE", 3),
        (Default, {}, "default_path", "PAY_0", 12),
        (Default, {}, "default_path", "PAY_3", 12),
        (Default, {}, "default_path", "LIMIT_BAL", 1),
        (Default, {}, "default_path", "PAY_AMT5", 1),
        (Default, {"raw": True}, "default_raw_path", "SEX", 1),
        (Default, {"raw": True}, "default_raw_path", "EDUCATION", 1),
        (Default, {"raw": True}, "default_raw_path", "MARRIAGE", 1),
        (Default, {"raw": True}, "default_raw_path", "PAY_0", 1),
        (Default, {"raw": True}, "default_raw_path", "PAY_3", 1),
        (Default, {"raw": True}, "default_raw_path", "LIMIT_BAL", 1),
        (Default, {"raw": True}, "default_raw_path", "PAY_AMT5", 1),
        (LawSchool, {}, "law_school_path", "race1", 5),
        (LawSchool, {}, "law_school_path", "gender", 2),
        (LawSchool, {}, "law_school_path", "lsat", 1),
        (LawSchool, {}, "law_school_path", "zgpa", 1),
        (LawSchool, {"raw": True}, "law_school_raw_path", "race1", 1),
        (LawSchool, {"raw": True}, "law_school_raw_path", "gender", 1),
        (LawSchool, {"raw": True}, "law_school_raw_path", "lsat", 1),
        (LawSchool, {"raw": True}, "law_school_raw_path", "zgpa", 1),
        (SouthGerman, {}, "south_german_path", "credit_history", 5),
        (SouthGerman, {}, "south_german_path", "personal_status_sex", 4),
        (SouthGerman, {}, "south_german_path", "foreign_worker", 2),
        (SouthGerman, {}, "south_german_path", "age", 1),
        (SouthGerman, {"raw": True}, "south_german_raw_path", "credit_history", 1),
        (SouthGerman, {"raw": True}, "south_german_raw_path", "personal_status_sex", 1),
        (SouthGerman, {"raw": True}, "south_german_raw_path", "foreign_worker", 1),
        (SouthGerman, {"raw": True}, "south_german_raw_path", "age", 1),
    ],
)
def test_column_indices_default(
    dataset_class,
    init_kwargs,
    dataset_path,
    variable,
    expected_num_columns,
    request,
):
    dataset_path = request.getfixturevalue(dataset_path)
    dataset = dataset_class(root=dataset_path, **init_kwargs)
    assert len(dataset.column_indices(variable)) == expected_num_columns
