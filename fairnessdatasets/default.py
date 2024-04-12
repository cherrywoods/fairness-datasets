# Copyright (c) 2024 Eric Tepper, David Boetius
# Licensed under the MIT license
from typing import Callable, Dict, Optional, Tuple, Union

import os

import pandas

from .base import DefaultPreprocessing


class Default(DefaultPreprocessing):
    """
    The `Default of credit card clients <https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients>`_
    dataset (Default for short).

    The dataset is preprocessed by:
     - Remove the "ID" column.
     - one-hot encoding all categorical attributes
     - applying z-score normalization to all continuous variables
    The last two preprocessing steps are optional and can be turned off by
    passing :code:`raw=True` to the initializer.

    Class Attributes:
    - `dataset_url`: The URL the Default dataset is downloaded from.
    - `file_to_download`: The file to download from `dataset_url`.
    - `checksums`: The checksums of the files to download from `dataset_url`.
    - `variables`: The variables of the Default dataset,
       together with the values they may take on and the interpretation of the values.

    Attributes:
    - `columns`: Column labels for the tensors in this dataset
      (after one-hot encoding, if applied).
    - `files_dir`: Where the data files are stored or downloaded to.
      Value: Root path (user specified) / `type(self).__name__)`
    - `data`: The dataset features (x values)
    - `targets` The dataset targets (y values)
    """

    dataset_url = "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip"
    file_to_download = "default of credit card clients.xls"
    checksums = {
        "default of credit card clients.xls": "30c6be3abd8dcfd3e6096c828bad8c2f011238620f5369220bd60cfc82700933",
    }

    PAYMENT_VALUES = {
        -2: "-2",
        -1: "pay duly",
        0: "0",
        1: "payment delay for one month",
        2: "payment delay for two months",
        3: "payment delay for three months",
        4: "payment delay for four months",
        5: "payment delay for five months",
        6: "payment delay for six months",
        7: "payment delay for seven months",
        8: "payment delay for eight months",
        9: "payment delay for nine months and above",
    }
    variables = {
        "LIMIT_BAL": None,
        "SEX": {
            1: "male",
            2: "female",
        },
        "EDUCATION": {
            1: "graduate school",
            2: "university",
            3: "high school",
            4: "others",
        },
        "MARRIAGE": {
            1: "married",
            2: "single",
            3: "others",
        },
        "AGE": None,
        "PAY_0": PAYMENT_VALUES,
        "PAY_2": PAYMENT_VALUES,
        "PAY_3": PAYMENT_VALUES,
        "PAY_4": PAYMENT_VALUES,
        "PAY_5": PAYMENT_VALUES,
        "PAY_6": PAYMENT_VALUES,
        "BILL_AMT1": None,
        "BILL_AMT2": None,
        "BILL_AMT3": None,
        "BILL_AMT4": None,
        "BILL_AMT5": None,
        "BILL_AMT6": None,
        "PAY_AMT1": None,
        "PAY_AMT2": None,
        "PAY_AMT3": None,
        "PAY_AMT4": None,
        "PAY_AMT5": None,
        "PAY_AMT6": None,
    }
    _variables_no_desc = {
        variable: None if values is None else tuple(values)
        for variable, values in variables.items()
    }

    def __init__(
        self,
        root: Union[str, os.PathLike],
        raw: bool = False,
        download: bool = False,
        output_fn: Optional[Callable[[str], None]] = print,
    ):
        """
        Loads the `Default <https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients>`_
        dataset.

        :param root: The root directory where the Default folder is placed or
          is to be downloaded to if download is set to True.
        :param raw: When :code:`True`, no one-hot encoding and standardization
         is applied to the downloaded data.
        :param download: Whether to download the Default of credit card clients dataset from
          https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients if it is not
          present in the root directory.
        :param output_fn: A function for producing command line output.
          For example, :code:`print` or :code:`logging.info`.
          Pass `None` to turn off command line output.
        """
        super().__init__(root, raw, download, output_fn)

    def _raw_files(self) -> tuple[str, ...]:
        return (self.file_to_download,)

    def _target_column(self) -> str:
        return "default payment next month"

    def _download(self):
        self._download_zip(self.dataset_url, self.checksums)

    def _load_raw(self) -> Tuple[pandas.DataFrame]:
        columns = list(self.variables.keys()) + [self._target_column()]
        data: pandas.DataFrame = pandas.read_excel(
            self.files_dir / self.file_to_download,
            sheet_name=0,
            header=None,
            names=columns,
            skiprows=2,
            index_col=0,
            dtype=int,
        )
        return (data,)

    def _preprocess(self, *data: pandas.DataFrame) -> Tuple[pandas.DataFrame]:
        self._output("Preprocessing data...")
        data = super()._preprocess(*data)
        self._output("Preprocessing finished.")
        return data

    def _variables(self) -> Dict[str, Optional[Tuple[str, ...]]]:
        return self._variables_no_desc
