# Copyright (c) 2023 David Boetius
# Licensed under the MIT license
from tempfile import TemporaryDirectory

import pytest

from fairnessdatasets import Adult, SouthGerman, Default, LawSchool


@pytest.fixture(scope="session")
def adult_path():
    with TemporaryDirectory() as tmp_dir:
        Adult(tmp_dir, train=False, download=True)
        yield tmp_dir


@pytest.fixture(scope="session")
def adult_raw_path(adult_path):
    Adult(adult_path, train=False, raw=True, download=True)
    yield adult_path


@pytest.fixture(scope="session")
def default_path():
    with TemporaryDirectory() as tmp_dir:
        Default(tmp_dir, download=True)
        yield tmp_dir


@pytest.fixture(scope="session")
def default_raw_path(default_path):
    Default(default_path, raw=True, download=True)
    yield default_path


@pytest.fixture(scope="session")
def law_school_path():
    with TemporaryDirectory() as tmp_dir:
        LawSchool(tmp_dir, download=True)
        yield tmp_dir


@pytest.fixture(scope="session")
def law_school_raw_path(default_path):
    LawSchool(default_path, raw=True, download=True)
    yield default_path


@pytest.fixture(scope="session")
def south_german_path():
    with TemporaryDirectory() as tmp_dir:
        SouthGerman(tmp_dir, download=True)
        yield tmp_dir


@pytest.fixture(scope="session")
def south_german_raw_path(south_german_path):
    SouthGerman(south_german_path, raw=True, download=True)
    yield south_german_path
