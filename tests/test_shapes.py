# Copyright (c) 2024 David Boetius
# Licensed under the MIT license
import pytest

from fairnessdatasets import Adult, SouthGerman, Default, LawSchool


@pytest.mark.parametrize(
    "dataset_class,init_kwargs,dataset_path,expected_data_shape,expected_targets_shape",
    [
        (Adult, {}, "adult_path", (30162, 104), (30162,)),
        (Adult, {"train": False}, "adult_path", (15060, 104), (15060,)),
        (Adult, {"raw": True}, "adult_raw_path", (30162, 14), (30162,)),
        (Adult, {"train": False, "raw": True}, "adult_raw_path", (15060, 14), (15060,)),
        (Default, {}, "default_path", (30000, 95), (30000,)),
        (Default, {"raw": True}, "default_raw_path", (30000, 23), (30000,)),
        (LawSchool, {}, "law_school_path", (20888, 10), (20888,)),
        (LawSchool, {"raw": True}, "law_school_raw_path", (20888, 5), (20888,)),
        (SouthGerman, {"raw": False}, "south_german_path", (1000, 72), (1000,)),
        (SouthGerman, {"raw": True}, "south_german_raw_path", (1000, 20), (1000,)),
    ],
    ids=[
        "Adult-train",
        "Adult-test",
        "Adult-raw-train",
        "Adult-raw-test",
        "Default",
        "Default-raw",
        "LawSchool",
        "LawSchool-raw",
        "SouthGerman",
        "SouthGerman-raw",
    ],
)
def test_data_shapes(
    dataset_class,
    init_kwargs,
    dataset_path,
    expected_data_shape,
    expected_targets_shape,
    request,
):
    dataset_path = request.getfixturevalue(dataset_path)
    dataset = dataset_class(root=dataset_path, **init_kwargs)
    assert dataset.data.shape == expected_data_shape
    assert dataset.targets.shape == expected_targets_shape
