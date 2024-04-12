# Copyright (c) 2024 David Boetius
# Licensed under the MIT license
from typing import Callable, Dict, Optional, Tuple, Union

import os

import pandas

from .base import DefaultPreprocessing


class LawSchool(DefaultPreprocessing):
    """
    The `Law School Admissions <https://eric.ed.gov/?id=ED469370>` dataset
    (Law School for short), downloaded from
    https://storage.googleapis.com/lawschool_dataset/bar_pass_prediction.csv.

    The dataset is preprocessed by:
     - Removing all columns, except for "race", "gender",
       "LSAT" (Law School Admission Test), "ZGPA" (Grade-Point Average),
       and "ZFYGPA" (First-Year Grade-Point Average).
       The "Z" presumably stands for these columns being z-score normalized.
     - removing rows (samples) with missing values
     - one-hot encoding all categorical attributes
     - applying z-score normalization to all continuous variables
    These preprocessing steps are optional and can be turned off by
    passing :code:`raw=True` to the initializer.

    Class Attributes:
    - `dataset_url`: The URL the Default dataset is downloaded from.
    - `checksum`: The checksum of the file to download from `dataset_url`.
    - `variables`: The selection of variables from the dataset that this class uses.
       Each variable is accompanied by the values which it can take on.
       For continuous variables, this entry is :code:`None`.

    Attributes:
    - `columns`: Column labels for the tensors in this dataset
      (after one-hot encoding, if applied).
    - `files_dir`: Where the data files are stored or downloaded to.
      Value: Root path (user specified) / `type(self).__name__)`
    - `data`: The dataset features (x values)
    - `targets` The dataset targets (y values)
    """

    dataset_url = (
        "https://storage.googleapis.com/lawschool_dataset/bar_pass_prediction.csv"
    )
    checksum = "c9a1d27b932bd697641c1c848d7e965232910285307e2edeb01c71edaeef11a2"

    variables = {
        "lsat": None,
        "zgpa": None,
        "zfygpa": None,
        "gender": ("female", "male"),
        "race1": ("black", "asian", "hisp", "white", "other"),
    }

    def __init__(
        self,
        root: Union[str, os.PathLike],
        raw: bool = False,
        download: bool = False,
        output_fn: Optional[Callable[[str], None]] = print,
    ):
        """
        Loads the `Law School <https://eric.ed.gov/?id=ED469370>`_ dataset.

        :param root: The root directory where the Default folder is placed or
          is to be downloaded to if download is set to True.
        :param raw: When :code:`True`, no one-hot encoding and standardization
         is applied to the downloaded data.
        :param download: Whether to download the Default of credit card clients dataset from
          https://storage.googleapis.com/lawschool_dataset/bar_pass_prediction.csv
          if it is not present in the root directory.
        :param output_fn: A function for producing command line output.
          For example, :code:`print` or :code:`logging.info`.
          Pass `None` to turn off command line output.
        """
        super().__init__(root, raw, download, output_fn)

    def _target_column(self) -> str:
        return "pass_bar"

    def _download(self):
        self._download_file(self.dataset_url, "raw.csv", self.checksum)

    def _load_raw(self) -> Tuple[pandas.DataFrame]:
        data: pandas.DataFrame = pandas.read_csv(
            self.files_dir / "raw.csv",
            header=0,
            index_col=0,
        )
        columns = list(self.variables.keys()) + [self._target_column()]
        return (data.filter(columns),)

    def _preprocess(self, *data: pandas.DataFrame) -> Tuple[pandas.DataFrame]:
        self._output("Preprocessing data...")
        data = self._strip_strings(*data)
        # remove rows with missing values
        data[0].dropna(axis=0, inplace=True)

        data = super()._preprocess(*data)
        self._output("Preprocessing finished.")
        return data

    def _variables(self) -> Dict[str, Optional[Tuple[str, ...]]]:
        return self.variables
