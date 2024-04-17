# Copyright (c) 2024 David Boetius
# Licensed under the MIT license
import itertools
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
    - `data`: The dataset features (x values)
    - `targets` The dataset targets (y values)
    """

    def __init__(
        self,
        root: Union[str, os.PathLike],
        download: bool = False,
        output_fn: Optional[Callable[[str], None]] = print,
    ):
        """
        Creates a new :code:`FairnessDataset`.

        :param root: The root directory containing the data.
         If :code:`download=True` and the data isn't present in :code:`root`,
         it is downloaded to :code:`root`.
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
        if any(not (self.files_dir / file).exists() for file in self._raw_files()):
            if not download:
                raise RuntimeError(
                    "Dataset not found. Download it by passing download=True."
                )
            os.makedirs(self.files_dir, exist_ok=True)
            self._download()

        if any(not (self.files_dir / file).exists() for file in self._data_files()):
            data = self._load_raw()
            data = self._preprocess(*data)
            self._store_preprocessed(*data)

        table = self._load_preprocessed(self._data_file())
        data = table.drop(self._target_column(), axis=1)
        targets = table[self._target_column()]

        self.data = torch.tensor(
            data.values.astype(np.float64), dtype=torch.get_default_dtype()
        )
        self.targets = torch.tensor(targets.values.astype(np.int64))

    def _raw_files(self) -> tuple[str, ...]:
        """
        The downloaded files before preprocessing.

        Used for checking whether the dataset was already downloaded.
        """
        return ("raw.csv",)

    def _data_files(self) -> tuple[str, ...]:
        """
        The preprocessed data files.

        Used for checking whether the dataset is available in preprocessed form.
        """
        return ("data.csv",)

    def _data_file(self) -> str:
        """
        The data file containing the data of this instance
        (e.g. training data file vs test data file).

        Returns "data.csv" by default.
        """
        return "data.csv"

    def _target_column(self) -> str:
        """
        The column in the data csv containing the target variable.

        Default value: `"target"`
        """
        return "target"

    @abstractmethod
    def _download(self):
        """
        Download and extract the dataset to :code:`self.files_dir` so that
        the raw data can be loaded using :code:`self._load_raw()`.
        """
        raise NotImplementedError()

    def _load_raw(self) -> Tuple[pandas.DataFrame, ...]:
        """
        Load the raw data (before preprocessing) from :code:`self.files_dir`.
        By default, loads the files in :code:`self._raw_files` as csv files.

        :return: The raw data as :code:`pandas.DataFrames`.
        """
        return tuple(
            pandas.read_csv(self.files_dir / file) for file in self._raw_files()
        )

    @abstractmethod
    def _preprocess(self, *data: pandas.DataFrame) -> Tuple[pandas.DataFrame, ...]:
        """
        Preprocess downloaded data.

        :param data: The downloaded data.
        :return: The preprocessed data as :code:`pandas.DataFrames`.
        """
        raise NotImplementedError()

    def _store_preprocessed(self, *data: pandas.DataFrame):
        """
        Stores preprocessed data.

        The default implementation assumes that the order of :code:`data` matches
        the order of :code:`self._data_files` and stores the data in the corresponding
        files in the CSV format.
        Overwrite if this does not match your needs.
        """
        for file, table in zip(self._data_files(), data):
            table.to_csv(self.files_dir / file, index=False)

    def _load_preprocessed(self, file):
        """
        Loads preprocessed data.

        The default implementation loads :code:`file` as a csv.

        :param file: The file name of the data to load.
        """
        return pandas.read_csv(Path(self.files_dir, file))

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

    def _download_file(self, file_url: str, file_name: str, file_checksum: str):
        """
        Download a single file.

        :param file_name: The name of the file in which to store the download file.
        """
        self._output(f"Downloading {type(self).__name__} data...")
        file_path = self.files_dir / file_name
        file_path.touch(exist_ok=False)
        result = requests.get(file_url, stream=True)
        with open(file_path, "wb") as out_file:
            for chunk in result.iter_content(chunk_size=256):
                out_file.write(chunk)

        self._output("Checking integrity of downloaded files...")
        downloaded_file_checksum = self._sha256sum(file_path)
        if file_checksum != downloaded_file_checksum:
            raise RuntimeError(
                f"Downloaded file has different checksum than expected: {file_name}. "
                f"Expected sha256 checksum: {file_checksum}"
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
        *data: pandas.DataFrame, variables: Dict[str, Tuple[str, ...]], columns=None
    ) -> Tuple[pandas.DataFrame, ...]:
        """
        Applies a one-hot encoding to categorical variables.

        :param data: The tables to preprocess.
        :param variables: The categorical variables of the data as a mapping from
         variable names to variable values.
        :param columns: An optional set of columns that the tables should posses
         after preprocessing.
         Allows for adding columns for values that do not appear in the dataset.
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
        if columns is None:
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


class DefaultPreprocessing(CSVDataset, ABC):
    """
    A dataset that applies a one-hot encoding to all categorical variables
    and standardizes (z-score normalizes) the continuous variables.
    The class allows turning this normalization of by passing :code:`raw=True` to
    the initializer.

    Provides the :code:`columns` attribute and
    """

    def __init__(
        self,
        root: Union[str, os.PathLike],
        raw: bool = False,
        download: bool = False,
        output_fn: Optional[Callable[[str], None]] = print,
    ):
        """
        Creates a new :code:`NormalizedDataset`.
        Setting :code:`raw=True` turns off normalization.
        """
        self._raw = raw

        if not self._raw:
            self.columns = tuple(
                itertools.chain(
                    *(
                        [col_name]
                        if values is None
                        else [f"{col_name}={value}" for value in values]
                        for col_name, values in self._variables().items()
                    )
                )
            )
        else:
            self.columns = tuple(col_name for col_name in self._variables())

        super().__init__(root, download, output_fn)

    def column_indices(self, variable: str) -> Tuple[int, ...]:
        assert variable in self._variables()
        return tuple(
            i for i, col in enumerate(self.columns) if col.startswith(variable)
        )

    @abstractmethod
    def _variables(self) -> Dict[str, Optional[Tuple[str, ...]]]:
        """
        The variables and variable values of the dataset.
        Return a mapping from variable names to variable values.
        For non-categorical variables, the variable value is :code:`None`.
        For categorical variables, the variable value is a tuple of strings.
        """
        raise NotImplementedError()

    def _data_files(self) -> tuple[str, ...]:
        if not self._raw:
            return ("data.csv",)
        else:
            return ("data_raw.csv",)

    def _data_file(self) -> str:
        if not self._raw:
            return "data.csv"
        else:
            return "data_raw.csv"

    def _preprocess(self, *data: pandas.DataFrame) -> Tuple[pandas.DataFrame, ...]:
        categorical = {
            var: tuple(vals)
            for var, vals in self._variables().items()
            if vals is not None
        }
        continuous = [var for var, vals in self._variables().items() if vals is None]
        if not self._raw:
            data = self._encode_one_hot(
                *data, variables=categorical, columns=self.columns
            )
            data = self._standardize(
                *data, variables=continuous, reference_data=self._reference_data(*data)
            )
        else:
            data = self._categorical_to_integer(*data, variables=categorical)

        # reorder dataset columns and rename
        all_columns = list(self.columns) + [self._target_column()]
        return tuple(table[all_columns] for table in data)

    def _reference_data(self, *data):
        """
        Select the dataset to use for obtaining mean and standard deviation
        estimates for standardization.
        """
        return data[0]
