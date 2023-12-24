# Copyright (c) 2023 David Boetius
# Licensed under the MIT license
from tempfile import TemporaryDirectory

import pytest

from adult import Adult, AdultRaw


@pytest.fixture(scope="session")
def adult_path():
    with TemporaryDirectory() as tmp_dir:
        Adult(tmp_dir, train=False, download=True)
        yield tmp_dir


@pytest.fixture(scope="session")
def adult_raw_path():
    with TemporaryDirectory() as tmp_dir:
        AdultRaw(tmp_dir, train=False, download=True)
        yield tmp_dir
