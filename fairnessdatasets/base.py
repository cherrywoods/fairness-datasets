# Copyright (c) 2024 David Boetius
# Licensed under the MIT license
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Callable, Sequence, Tuple, Union

import os
from pathlib import Path

import numpy as np
import requests
import hashlib
from zipfile import ZipFile

import pandas
import torch
from torch.utils.data import Dataset


class CSVDataset(Dataset, ABC):
    """
    Base class for fairness datasets.
    Provides methods for downloading and processing data.

    Attributes:
        - `files_dir`: Where the data files are stored or downloaded to.
          Value: Root path (user specified) / `type(self).__name__)`
    """

    _train_file = "train.csv"
    _test_file = "test.csv"

    def __init__(
        self,
        root: Union[str, os.PathLike],
        train: bool = True,
        download: bool = False,
        output_fn: Optional[Callable[[str], None]] = print,
    ):
        """
        Creates a new :code:`FairnessDataset`.

        :param root: The root directory containing the data.
         If :code:`download=True` and the data isn't present in :code:`root`,
         it is downloaded to :code:`root`.
        :param train: Whether to retrieve the training set or test set of
          the dataset.
        :param download: Whether to download the dataset if it is not
          present in the :code:`root` directory.
        :param output_fn: A function for producing command line output.
          For example, :code:`print` or :code:`logging.info`.
          Pass `None` to turn off command line output.

        """
        if output_fn is None:

            def do_nothing(_):
                pass

            self.__output_fn = do_nothing
        else:
            self.__output_fn = output_fn

        self.files_dir = Path(root, type(self).__name__)
        if not self.files_dir.exists():
            if not download:
                raise RuntimeError(
                    "Dataset not found. Download it by passing download=True."
                )
            os.makedirs(self.files_dir)
            data = self._download()
            data = self._preprocess(*data)
            train_data, test_data = self._train_test_split(*data)

            # create new csv files for the transformed data
            train_data.to_csv(Path(self.files_dir, self._train_file), index=False)
            test_data.to_csv(Path(self.files_dir, self._test_file), index=False)

            if train:
                table = train_data
            else:
                table = test_data
        else:
            if train:
                table = pandas.read_csv(Path(self.files_dir, self._train_file))
            else:
                table = pandas.read_csv(Path(self.files_dir, self._test_file))

        data = table.drop(self._target_column(), axis=1)
        targets = table[self._target_column()]

        self.data = torch.tensor(
            data.values.astype(np.float64), dtype=torch.get_default_dtype()
        )
        self.targets = torch.tensor(targets.values.astype(np.int64))

    @abstractmethod
    def _download(self) -> Tuple[pandas.DataFrame]:
        """
        Download and extract the dataset to :code:`self.files_dir`.

        :return: The downloaded data as :code:`pandas.DataFrames`.
        """
        raise NotImplementedError()

    @abstractmethod
    def _preprocess(self, *data: pandas.DataFrame) -> Tuple[pandas.DataFrame, ...]:
        """
        Preprocess downloaded data.

        :param data: The downloaded data.
        :return: The preprocessed data as :code:`pandas.DataFrames`.
        """
        raise NotImplementedError()

    @abstractmethod
    def _train_test_split(
        self, *data: pandas.DataFrame
    ) -> Tuple[pandas.DataFrame, pandas.DataFrame]:
        """
        Split downloaded and preprocessed data into training and test sets.

        :param data: The downloaded data.
        :return: A training and a test set
        """
        raise NotImplementedError()

    @abstractmethod
    def _target_column(self) -> str:
        raise NotImplementedError()

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.data[index], self.targets[index]

    def __len__(self):
        return len(self.targets)

    def _download_zip(
        self,
        dataset_url: str,
        file_checksums: Dict[str, str],
    ):
        """
        Download and extract dataset .zip files.
        """
        self._output(f"Downloading {type(self).__name__} data...")
        dataset_path = self.files_dir / "dataset.zip"
        try:
            dataset_path.touch(exist_ok=False)
            result = requests.get(dataset_url, stream=True)
            with open(dataset_path, "wb") as dataset_file:
                for chunk in result.iter_content(chunk_size=256):
                    dataset_file.write(chunk)
            with ZipFile(dataset_path) as dataset_archive:
                for file_name in file_checksums:
                    dataset_archive.extract(file_name, self.files_dir)
        finally:
            dataset_path.unlink(missing_ok=True)

        self._output("Checking integrity of downloaded files...")
        for file_name, checksum in file_checksums.items():
            file_path = self.files_dir / file_name
            downloaded_file_checksum = self._sha256sum(file_path)
            if checksum != downloaded_file_checksum:
                raise RuntimeError(
                    f"Downloaded file has different checksum than expected: {file_name}. "
                    f"Expected sha256 checksum: {checksum}"
                )
        self._output("Download finished.")

    @staticmethod
    def _strip_strings(*data: pandas.DataFrame) -> Tuple[pandas.DataFrame, ...]:
        """Strips all strings in several :code:`DataFrames`."""
        return tuple(
            table.map(lambda val: val.strip() if isinstance(val, str) else val)
            for table in data
        )

    @staticmethod
    def _remove_missing_values(
        *data: pandas.DataFrame, marker="?"
    ) -> Tuple[pandas.DataFrame, ...]:
        """
        Removes rows with missing values from table.
        Modifies the data in-place.

        :param marker: The value marking missing values.
        :return: The preprocessed data (same as :code:`*data`).
        """
        for table in data:
            table.replace(to_replace=marker, value=np.nan, inplace=True)
            table.dropna(axis=0, inplace=True)
        return data

    @staticmethod
    def _categorical_to_integer(
        *data: pandas.DataFrame, variables: Sequence[str]
    ) -> Tuple[pandas.DataFrame, ...]:
        """
        Replaces string values of categorical attributes with integers.
        Modifies the data in-place.

        :param data: The tables to preprocess.
        :param variables: The categorical variables of the data as a mapping from
         variable names to variable values.
        :return: The preprocessed data (same as :code:`*data`).
        """
        for variable, values in variables.items():
            remapping = {value: index for index, value in enumerate(values)}
            for table in data:
                table.replace(remapping, inplace=True)
        return data

    @staticmethod
    def _encode_one_hot(
        *data: pandas.DataFrame, variables: Dict[str, Tuple[str, ...]]
    ) -> Tuple[pandas.DataFrame, ...]:
        """
        Applies a one-hot encoding to categorical variables.

        :param data: The tables to preprocess.
        :param variables: The categorical variables of the data as a mapping from
         variable names to variable values.
        :return: The preprocessed data.
        """
        tables = data

        # one-hot encode all categorical variables
        tables = tuple(
            pandas.get_dummies(table, columns=variables.keys(), prefix_sep="=")
            for table in tables
        )

        # some tables may not contain all values of a categorical variable
        # make sure all tables have the same columns
        columns = set()
        for table in tables:
            columns.update(table.columns)
        for col in columns:
            for table in tables:
                if col not in table.columns:
                    table.insert(loc=0, column=col, value=0.0)
        return tables

    @staticmethod
    def _standardize(
        *data: pandas.DataFrame,
        variables: Sequence[str],
        reference_data: pandas.DataFrame,
    ) -> Tuple[pandas.DataFrame, ...]:
        """
        Z-score normalizes (standardizes) continuous variables.
        Modifies the data in-place.

        :param data: The tables to preprocess.
        :param variables: The continuous variables
        :param reference_data: The data to use for computing means and
         standard deviations of the continuous variables.
        :return: The preprocessed data (same as :code:`*data`).
        """
        # standardise continuous columns (z score)
        for col in variables:
            mean = reference_data[col].mean()
            std = reference_data[col].std()
            for table in data:
                table[col] = (table[col] - mean) / std
        return data

    @staticmethod
    def _sha256sum(path):
        # based on: https://stackoverflow.com/a/3431838/10550998 by quantumSoup
        # License: CC-BY-SA
        hash_ = hashlib.sha256()
        with open(path, "rb") as file:
            for chunk in iter(lambda: file.read(4096), b""):
                hash_.update(chunk)
        return hash_.hexdigest()

    def _output(self, message: str):
        """Logging utility."""
        self.__output_fn(message)
