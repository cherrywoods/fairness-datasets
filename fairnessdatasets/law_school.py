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
     - Selecting several columns. By default, the columns
       "race1", "gender",
       "LSAT" (Law School Admission Test), "ZGPA" (normalized Grade-Point Average),
       and "ZFYGPA" (normalized First-Year Grade-Point Average (?)).
       The "Z" presumably stands for these columns being z-score normalized.
       All available columns are listed in the `LawSchool.variables` class attribute.
     - removing rows (samples) with missing values in the selected columns.
     - one-hot encoding all categorical attributes
     - applying z-score normalization to all continuous variables
    The last two preprocessing steps are optional and can be turned off by
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
        "decible1b": tuple(f"{i}" for i in range(1, 11)),
        "decible3": tuple(f"{i}" for i in range(1, 11)),
        "ID": None,
        "decible1": tuple(f"{i}" for i in range(1, 11)),
        # the "sex" and "gender" columns are identical
        "race": tuple(f"{i}" for i in range(1, 9)),
        "cluster": tuple(f"{i}" for i in range(1, 7)),
        "lsat": None,
        "ugpa": None,
        "zfygpa": None,
        "DOB_yr": None,
        "grad": ("Y", "X", "O"),
        "zgpa": None,
        "bar1": ("P", "F"),
        "bar1_yr": None,
        "bar2": ("P", "F"),
        "bar2_yr": None,
        "fulltime": ("1", "2"),
        "fam_inc": tuple(f"{i}" for i in range(1, 6)),
        "age": None,
        "gender": ("female", "male"),
        "parttime": ("0", "1"),
        "race1": ("black", "asian", "hisp", "white", "other"),
        "Dropout": ("NO", "YES"),
        "pass_bar": ("0", "1"),
        "bar": ("a Passed 1st time", "c Failed", "b Passed 2nd time", "e non-Grad"),
        "tier": tuple(f"{i}" for i in range(1, 7)),
        "index6040": None,
        "indxgrp": (
            "g 700+",
            "f 640-700",
            "e 580-640",
            "d 520-580",
            "c 460-520",
            "b 400-460",
            "a under 400",
        ),
        "indxgrp2": (
            "i 820+",
            "f 640-700",
            "h 760-820",
            "g 700-760",
            "e 580-640",
            "d 520-580",
            "c 460-520",
            "b 400-460",
            "a under 400",
        ),
        "gpa": None,
    }

    def __init__(
        self,
        root: Union[str, os.PathLike],
        features: Tuple[str, ...] = ("lsat", "zgpa", "zfygpa", "gender", "race1"),
        target: str = "pass_bar",
        raw: bool = False,
        download: bool = False,
        output_fn: Optional[Callable[[str], None]] = print,
    ):
        """
        Loads the `Law School <https://eric.ed.gov/?id=ED469370>`_ dataset.

        :param root: The root directory where the Default folder is placed or
          is to be downloaded to if download is set to True.
        :param features: The variables to include as features in the dataset.
        :param target: The target variable. This is the variable that a model
         learns to predict.
        :param raw: When :code:`True`, no one-hot encoding and standardization
         is applied to the downloaded data.
        :param download: Whether to download the Default of credit card clients dataset from
          https://storage.googleapis.com/lawschool_dataset/bar_pass_prediction.csv
          if it is not present in the root directory.
        :param output_fn: A function for producing command line output.
          For example, :code:`print` or :code:`logging.info`.
          Pass `None` to turn off command line output.
        """
        for var in features:
            if var not in LawSchool.variables:
                raise ValueError(f"Unknown variable {var}")
        if target not in LawSchool.variables:
            raise ValueError(f"Unknown variable {target}")
        self.__target = target
        self.__features = list(features)
        super().__init__(root, raw, download, output_fn)

    def _target_column(self) -> str:
        return self.__target

    def _download(self):
        self._download_file(self.dataset_url, "raw.csv", self.checksum)

    def _data_file(self) -> str:
        columns = "-".join(self.__features)
        return f"data-{columns}-{self.__target}.csv"

    def _data_files(self) -> tuple[str, ...]:
        return (self._data_file(),)

    def _load_raw(self) -> Tuple[pandas.DataFrame]:
        data: pandas.DataFrame = pandas.read_csv(
            self.files_dir / "raw.csv",
            header=0,
            index_col=0,
        )
        return (data[self.__features + [self.__target]],)

    def _preprocess(self, *data: pandas.DataFrame) -> Tuple[pandas.DataFrame]:
        self._output("Preprocessing data...")
        # select columns
        data = (data[0][self.__features + [self.__target]],)
        data = self._strip_strings(*data)
        # remove rows with missing values
        data[0].dropna(axis=0, inplace=True)

        data = super()._preprocess(*data)
        self._output("Preprocessing finished.")
        return data

    def _variables(self) -> Dict[str, Optional[Tuple[str, ...]]]:
        return {
            var: vals for var, vals in self.variables.items() if var in self.__features
        }
