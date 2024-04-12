# Copyright (c) 2024 Eric Tepper, David Boetius
# Licensed under the MIT license
from typing import Callable, Dict, Optional, Tuple, Union

import os

import pandas

from .base import DefaultPreprocessing


class SouthGerman(DefaultPreprocessing):
    """
    The `South German Credit <https://archive.ics.uci.edu/dataset/573/south+german+credit+update>`_ dataset.

    The dataset is preprocessed by:
     - one-hot encoding all categorical attributes
     - applying z-score normalization to all continuous variables
    These preprocessing steps are optional and can be turned off by
    passing :code:`raw=True` to the initializer.

    Class Attributes:
    - `dataset_url`: The URL the South German Credit dataset is downloaded from.
    - `file_to_download`: The file to download from `dataset_url`.
    - `checksums`: The checksums of the files to download from `dataset_url`.
    - `file`: The file containing the data after downloading.
    - `variables`: The variables of the South German Credit dataset,
       together with the values they may take on and the interpretation of the values.

    Attributes:
    - `columns`: Column labels for the tensors in this dataset
      (after one-hot encoding, if applied).
    - `files_dir`: Where the data files are stored or downloaded to.
      Value: Root path (user specified) / `type(self).__name__)`
    - `data`: The dataset features (x values)
    - `targets` The dataset targets (y values)
    """

    dataset_url = (
        "https://archive.ics.uci.edu/static/public/573/south+german+credit+update.zip"
    )
    file_to_download = "SouthGermanCredit.asc"
    checksums = {
        "SouthGermanCredit.asc": "5f363343f356ca38a0236baab849e472846399b2176ccc5bd686483dd8a7562f",
    }

    variables = {
        "status": {
            1: "no checking account",
            2: "... < 0 DM",
            3: "0<= ... < 200 DM",
            4: "... >= 200 DM",
        },
        "duration": None,
        "credit_history": {
            0: "delay in paying off in the past",
            1: "critical account/other credits elsewhere",
            2: "no credits taken/all credits paid back duly",
            3: "existing credits paid back duly till now",
            4: "all credits at this bank paid back duly",
        },
        "purpose": {
            0: "others",
            1: "car (new)",
            2: "car (used)",
            3: "furniture/equipment",
            4: "radio/television",
            5: "domestic appliances",
            6: "repairs",
            7: "education",
            8: "vacation",
            9: "retraining",
            10: "business",
        },
        "amount": None,
        "savings": {
            1: "unknown/no savings account",
            2: "... <  100 DM",
            3: "100 <= ... <  500 DM",
            4: "500 <= ... < 1000 DM",
            5: "... >= 1000 DM",
        },
        "employment_duration": {
            1: "unemployed",
            2: "< 1 yr",
            3: "1 <= ... < 4 yrs",
            4: "4 <= ... < 7 yrs",
            5: ">= 7 yrs",
        },
        "installment_rate": {
            1: ">= 35",
            2: "25 <= ... < 35",
            3: "20 <= ... < 25",
            4: "< 20",
        },
        "personal_status_sex": {
            1: "male : divorced/separated",
            2: "female : non-single or male : single",
            3: "male : married/widowed",
            4: "female : single",
        },
        "other_debtors": {
            1: "none",
            2: "co-applicant",
            3: "guarantor",
        },
        "present_residence": {
            1: "< 1 yr",
            2: "1 <= ... < 4 yrs",
            3: "4 <= ... < 7 yrs",
            4: ">= 7 yrs",
        },
        "property": {
            1: "unknown / no property",
            2: "car or other",
            3: "building soc. savings agr./life insurance",
            4: "real estate",
        },
        "age": None,
        "other_installment_plans": {
            1: "bank",
            2: "stores",
            3: "none",
        },
        "housing": {
            1: "for free",
            2: "rent",
            3: "own",
        },
        "number_credits": {
            1: "1",
            2: "2-3",
            3: "4-5",
            4: ">= 6",
        },
        "job": {
            1: "unemployed/unskilled - non-resident",
            2: "unskilled - resident",
            3: "skilled employee/official",
            4: "manager/self-empl./highly qualif. employee",
        },
        "people_liable": {
            1: "3 or more",
            2: "0 to 2",
        },
        "telephone": {
            1: "no",
            2: "yes (under customer name)",
        },
        "foreign_worker": {
            1: "yes",
            2: "no",
        },
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
        Loads the `South German Credit <https://archive.ics.uci.edu/dataset/573/south+german+credit+update>`_ dataset.

        :param root: The root directory where the SouthGerman folder is placed or
          is to be downloaded to if download is set to True.
        :param raw: When :code:`True`, no one-hot encoding and standardization
         is applied to the downloaded data.
        :param download: Whether to download the South German Credit dataset from
          https://archive.ics.uci.edu/dataset/573/south+german+credit+update if it is not
          present in the root directory.
        :param output_fn: A function for producing command line output.
          For example, :code:`print` or :code:`logging.info`.
          Pass `None` to turn off command line output.
        """
        super().__init__(root, raw, download, output_fn)

    def _raw_files(self) -> tuple[str, ...]:
        return (self.file_to_download,)

    def _target_column(self) -> str:
        return "credit_risk"

    def _download(self):
        self._download_zip(self.dataset_url, self.checksums)

    def _load_raw(self) -> Tuple[pandas.DataFrame]:
        all_columns = list(self.variables.keys()) + [self._target_column()]
        data: pandas.DataFrame = pandas.read_csv(
            self.files_dir / self.file_to_download,
            index_col=False,
            names=all_columns,
            delimiter=" ",
            skiprows=1,  # skip first row since pandas does not recognize header row
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
