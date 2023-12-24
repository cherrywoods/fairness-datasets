# Copyright (c) 2023 David Boetius
# Licensed under the MIT license
from typing import Tuple
from functools import reduce

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
    """

    categorical_features_map = reduce(
        lambda d1, d2: d1 | d2,
        (
            {value: index for index, value in enumerate(values)}
            for values in Adult.columns_with_values.values()
            if values is not None
        )
    )

    def _preprocess_features(
        self, train_data: pandas.DataFrame, test_data: pandas.DataFrame
    ) -> Tuple[pandas.DataFrame, pandas.DataFrame]:
        """
        Replaces string values of categorical attributes with integers.
        """
        for table in [train_data, test_data]:
            table.replace(self.categorical_features_map, inplace=True)
        return train_data, test_data
