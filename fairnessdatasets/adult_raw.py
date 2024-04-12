# Copyright (c) 2024 David Boetius
# Licensed under the MIT license
from typing import Tuple

import pandas

from .adult import Adult


class AdultRaw(Adult):
    """
    The `Adult <https://archive.ics.uci.edu/dataset/2/adult>`_ dataset.

    In difference, to the :code:`Adult` class, this class does not
    one-hot encode categorical attributes or normalize the continuous
    variables.
    However, it also removes rows with missing values.
    The categorical variables are instead encoded as integers.

    Attributes:
    The attributes are as for :code:`Adult`, except that the value
    of `AdultRaw.columns` has a different value.
    In particular, :code:`AdultRaw.columns = ("age", "workclass", "fnlwgt", ...)`
    """

    columns = tuple(col_name for col_name in Adult.variables)

    def _preprocess_features(
        self, train_data: pandas.DataFrame, test_data: pandas.DataFrame
    ) -> Tuple[pandas.DataFrame, pandas.DataFrame]:
        """
        Replaces string values of categorical attributes with integers.
        """
        categorical = {
            var: vals for var, vals in self.variables.items() if vals is not None
        }
        train_data, test_data = self._categorical_to_integer(
            train_data, test_data, variables=categorical
        )
        return train_data, test_data
