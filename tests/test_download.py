# Copyright (c) 2024 David Boetius
# Licensed under the MIT license
import pytest

from fairnessdatasets import Adult, SouthGerman, Default, LawSchool


@pytest.mark.parametrize(
    "dataset_class,init_kwargs",
    [
        (Adult, {}),
        (Adult, {"train": False}),
        (Adult, {"raw": True}),
        (Adult, {"train": False, "raw": True}),
        (Default, {}),
        (Default, {"raw": True}),
        (LawSchool, {}),
        (LawSchool, {"raw": True}),
        (SouthGerman, {"raw": False}),
        (SouthGerman, {"raw": True}),
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
def test_download(dataset_class, init_kwargs, tmp_path):
    dataset = dataset_class(root=tmp_path, download=True, **init_kwargs)
    assert len(dataset[0]) == 2


@pytest.mark.parametrize(
    "dataset_class",
    [Adult, SouthGerman, Default],
)
def test_download_alongside_raw(dataset_class, tmp_path):
    raw = dataset_class(root=tmp_path, raw=True, download=True)
    assert len(raw[0]) == 2
    standard = dataset_class(root=tmp_path, raw=False, download=True)
    assert len(standard[0]) == 2

    assert len(raw.columns) < len(standard.columns)


@pytest.mark.parametrize("raw", [False, True], ids=["default", "raw"])
def test_download_south_german(raw: bool, tmp_path):
    dataset = SouthGerman(root=tmp_path, raw=raw, download=True)
    assert len(dataset[0]) == 2


def test_download_string_dir(tmp_path):
    dataset = SouthGerman(root=str(tmp_path), download=True)
    assert len(dataset[0]) == 2
