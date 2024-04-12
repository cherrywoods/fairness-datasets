# Copyright (c) 2024 David Boetius
# Licensed under the MIT license
import itertools
from typing import Callable, Optional, Tuple, Union

import os

import pandas

from .base import CSVDataset


class Adult(CSVDataset):
    """
    The `Adult <https://archive.ics.uci.edu/dataset/2/adult>`_ dataset.

    The dataset is preprocessed by:
     - removing rows (samples) with missing values
     - one-hot encoding all categorical attributes
     - applying z-score normalization to all continuous variables

    Attributes:
     - `dataset_url`: The URL the Adult dataset is downloaded from.
     - `files_to_download`: The files that are downloaded from `dataset_url`.
     - `checksums`: The checksums of the files in `files`.
     - `train_file`: The file containing the training data after downloading.
     - `test_file`: The file containing the test data after downloading.
     - `variables`: The variables of the Adult dataset,
        together with the values they may take on.
        For integer variables, such as `age`, the value is :code:`None`.
        That is, :code:`Adult.variables["age"] = None`.
        For categorical variables, like `sex`, the value is a tuple of strings.
        For example, :code:`Adult.variables["sex"] = ("female", "male")`.
        Values that do not appear in the dataset after preprocessing are not
        included.
        This only affects the values of `workclass`, where `workclass=Never-worked`
        is dropped, as it does not appear in the dataset after dropping rows with
        missing values.
     - `columns`: Column labels for the tensors in this dataset (after one-hot encoding).
        This is :code:`Adult.columns = ("age", "workclass=Private",
        "workclass=Self-emp-not-inc", ...)`.
    """

    dataset_url = "https://archive.ics.uci.edu/static/public/2/adult.zip"
    files_to_download = {"test": "adult.test", "train": "adult.data"}
    checksums = {
        "adult.test": "a2a9044bc167a35b2361efbabec64e89d69ce82d9790d2980119aac5fd7e9c05",
        "adult.data": "5b00264637dbfec36bdeaab5676b0b309ff9eb788d63554ca0a249491c86603d",
    }
    train_file = "train.csv"
    test_file = "test.csv"
    variables = {
        "age": None,  # continuous variables marked with None
        "workclass": (
            "Private",
            "Self-emp-not-inc",
            "Self-emp-inc",
            "Federal-gov",
            "Local-gov",
            "State-gov",
            "Without-pay",
            # "Never-worked",  # does not appear in dataset
        ),
        "fnlwgt": None,
        "education": (
            "Bachelors",
            "Some-college",
            "11th",
            "HS-grad",
            "Prof-school",
            "Assoc-acdm",
            "Assoc-voc",
            "9th",
            "7th-8th",
            "12th",
            "Masters",
            "1st-4th",
            "10th",
            "Doctorate",
            "5th-6th",
            "Preschool",
        ),
        "education-num": None,
        "marital-status": (
            "Married-civ-spouse",
            "Divorced",
            "Never-married",
            "Separated",
            "Widowed",
            "Married-spouse-absent",
            "Married-AF-spouse",
        ),
        "occupation": (
            "Tech-support",
            "Craft-repair",
            "Other-service",
            "Sales",
            "Exec-managerial",
            "Prof-specialty",
            "Handlers-cleaners",
            "Machine-op-inspct",
            "Adm-clerical",
            "Farming-fishing",
            "Transport-moving",
            "Priv-house-serv",
            "Protective-serv",
            "Armed-Forces",
        ),
        "relationship": (
            "Wife",
            "Own-child",
            "Husband",
            "Not-in-family",
            "Other-relative",
            "Unmarried",
        ),
        "race": ("White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black"),
        "sex": ("Female", "Male"),
        "capital-gain": None,
        "capital-loss": None,
        "hours-per-week": None,
        "native-country": (
            "United-States",
            "Cambodia",
            "England",
            "Puerto-Rico",
            "Canada",
            "Germany",
            "Outlying-US(Guam-USVI-etc)",
            "India",
            "Japan",
            "Greece",
            "South",
            "China",
            "Cuba",
            "Iran",
            "Honduras",
            "Philippines",
            "Italy",
            "Poland",
            "Jamaica",
            "Vietnam",
            "Mexico",
            "Portugal",
            "Ireland",
            "France",
            "Dominican-Republic",
            "Laos",
            "Ecuador",
            "Taiwan",
            "Haiti",
            "Columbia",
            "Hungary",
            "Guatemala",
            "Nicaragua",
            "Scotland",
            "Thailand",
            "Yugoslavia",
            "El-Salvador",
            "Trinadad&Tobago",
            "Peru",
            "Hong",
            "Holand-Netherlands",
        ),
    }

    columns = tuple(
        itertools.chain(
            *(
                [col_name]
                if values is None
                else [f"{col_name}={value}" for value in values]
                for col_name, values in variables.items()
            )
        )
    )

    def __init__(
        self,
        root: Union[str, os.PathLike],
        train: bool = True,
        download: bool = False,
        output_fn: Optional[Callable[[str], None]] = print,
    ):
        """
        Loads the `Adult <https://archive.ics.uci.edu/dataset/2/adult>`_ dataset.

        :param root: The root directory where the Adult folder is placed or
          is to be downloaded to if download is set to True.
        :param train: Whether to retrieve the training set or test set of
          the dataset.
        :param download: Whether to download the Adult dataset from
          https://archive.ics.uci.edu/dataset/2/adult if it is not
          present in the root directory.
        :param output_fn: A function for producing command line output.
          For example, :code:`print` or :code:`logging.info`.
          Pass `None` to turn off command line output.
        """
        super().__init__(root, train, download, output_fn)

    def _download(self) -> Tuple[pandas.DataFrame, pandas.DataFrame]:
        """
        Downloads the Adult dataset, extracts the data and returns
        the raw training data and the raw test data as pandas :code:`DataFrames`.
        """
        self._download_zip(self.dataset_url, self.checksums)

        all_columns = list(self.variables.keys()) + ["income"]
        train_data: pandas.DataFrame = pandas.read_csv(
            self.files_dir / self.files_to_download["train"],
            header=None,
            index_col=False,
            names=all_columns,
        )
        test_data: pandas.DataFrame = pandas.read_csv(
            self.files_dir / self.files_to_download["test"],
            header=0,  # first row contains a note that we throw away
            index_col=False,
            names=all_columns,
        )
        return train_data, test_data

    _label_map = {"<=50K": False, ">50K": True, "<=50K.": False, ">50K.": True}

    def _preprocess(
        self, train_raw: pandas.DataFrame, test_raw: pandas.DataFrame
    ) -> Tuple[pandas.DataFrame, pandas.DataFrame]:
        self._output("Preprocessing data...")
        # preprocessing closely follows:
        # https://github.com/eth-sri/lcifr/blob/master/code/datasets/adult.py
        train_data, test_data = self._strip_strings(train_raw, test_raw)

        # transform the dataset: drop rows with missing values
        # missing values are encoded as ? in the original tables
        train_data, test_data = self._remove_missing_values(train_data, test_data)

        # map labels to (uniform) boolean values
        train_data.replace(self._label_map, inplace=True)
        test_data.replace(self._label_map, inplace=True)

        train_data, test_data = self._preprocess_features(train_data, test_data)

        all_columns = list(self.columns) + ["income"]
        train_data = train_data[all_columns]
        test_data = test_data[all_columns]

        self._output("Preprocessing finished.")
        return train_data, test_data

    def _preprocess_features(
        self, train_data: pandas.DataFrame, test_data: pandas.DataFrame
    ) -> Tuple[pandas.DataFrame, pandas.DataFrame]:
        categorical = {
            var: vals for var, vals in self.variables.items() if vals is not None
        }
        train_data, test_data = self._encode_one_hot(
            train_data, test_data, variables=categorical
        )
        continuous = [var for var, vals in self.variables.items() if vals is None]
        train_data, test_data = self._standardize(
            train_data, test_data, variables=continuous, reference_data=train_data
        )
        return train_data, test_data

    def _train_test_split(
        self, *data: pandas.DataFrame
    ) -> Tuple[pandas.DataFrame, pandas.DataFrame]:
        return data[0], data[1]

    def _target_column(self) -> str:
        return "income"

    def column_indices(self, variable: str) -> Tuple[int, ...]:
        assert variable in self.variables
        return tuple(
            i for i, col in enumerate(self.columns) if col.startswith(variable)
        )
