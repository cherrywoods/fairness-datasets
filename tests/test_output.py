# Copyright (c) 2023 David Boetius
# Licensed under the MIT license
import logging

from adult import Adult


def test_print(tmp_path):
    Adult(root=tmp_path, train=False, download=True, output_fn=print)


def test_logging_info(tmp_path, caplog):
    caplog.set_level(logging.INFO)
    Adult(root=tmp_path, train=False, download=True, output_fn=logging.info)
    assert len(caplog.records) > 0


def test_no_output(tmp_path, caplog):
    caplog.set_level(logging.INFO)
    Adult(root=tmp_path, train=False, download=True, output_fn=None)
    assert len(caplog.records) == 0
