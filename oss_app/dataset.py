import pandas as pd
import numpy as np
import pprint
import os
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler


# %% Classes


# Dataset class for handling input data initialization and processing
class Dataset:
    def __init__(
        self, raw_df: pd.DataFrame, prefiltered: bool = False, parameters: dict = None, properties: dict = None
    ):
        self.raw_df = raw_df
        self.filtered_df = None
        self.prefiltered = prefiltered
        self.parameters = parameters if parameters is not None else {}
        self.properties = properties if properties is not None else {}

    def initialize(self):
        """
        Initialize the dataset by setting up properties and filtering data further if needed.
        """
        self.properties["num_rows"] = len(self.raw_df)
        self.properties["num_columns"] = len(self.raw_df.columns)
        self.properties["column_names"] = list(self.raw_df.columns)

        # Optionally filter data based on parameters if not prefiltered
        if "filters" in self.parameters and not self.prefiltered:
            self.filter_data(**self.parameters["filters"])

        self.subject_id_variable = self.parameters.get("subject_id_variable", None)
        self.sex_variable = self.parameters.get("sex_variable", None)
        self.grouping_variable = self.parameters.get("grouping_variable", None)
        self.extra_index_variables = self.parameters.get("index_variables", [])
        if not self.parameters.get("indices", None):
            self.indices = [
                self.subject_id_variable,
                self.sex_variable,
                self.grouping_variable,
            ] + self.extra_index_variables
        else:
            self.indices = self.parameters["indices"]
        self.metric_variables = self.parameters.get("metric_variables", [])
        assert self.metric_variables, "No metric variables specified in parameters."

        # Fix column names for consistency
        self.fix_column_names()

    def filter_data(self, **kwargs):
        self.filtered_df = self.raw_df
        for key, value in kwargs.items():
            self.filtered_df = self.filtered_df[self.filtered_df[key] == value]
        return self.filtered_df

    def get_filtered_data(self):
        return self.filtered_df

    def fix_column_names(self):
        """
        Fix column names by removing leading/trailing whitespace and converting to lowercase.
        """
        self.raw_df.columns = [col.strip().lower() for col in self.raw_df.columns]
        if self.filtered_df is not None:
            self.filtered_df.columns = [col.strip().lower() for col in self.filtered_df.columns]

    def define_new_group(self, new_col_name: str, existing_cols: list):
        """
        Define a new group column based on unique combinations of existing columns.

        This is useful for creating a new categorical variable that combines multiple existing columns.
        Since filtering UI can handle multiple filters, this is mainly for labeling / exporting purposes.

        This function modifies both raw_df and filtered_df if it exists.

        Args:
            new_col_name (str): The name of the new column to be created.
            existing_cols (list): A list of column names to combine.
        """

        # Helper function to create the new column
        def create_group_col(df):
            for col in existing_cols:
                if col not in df.columns:
                    raise ValueError(f"Column '{col}' not found in the dataframe.")
            df[new_col_name] = df[existing_cols].astype(str).agg("_".join, axis=1)
            return df

        # Apply to raw_df
        self.raw_df = create_group_col(self.raw_df.copy())

        # Apply to filtered_df if it exists
        if self.filtered_df is not None:
            self.filtered_df = create_group_col(self.filtered_df.copy())

    def create_metric_from_operation(self, new_col_name: str, col1_name: str, col2_name: str, operation: callable):
        """
        Creates a new metric column by performing a mathematical operation on two existing columns.

        The operation is applied to both raw_df and filtered_df if it exists. The new column
        is added to the list of metric variables.

        Args:
            new_col_name (str): The name of the new column to be created.
            col1_name (str): The name of the first column for the operation.
            col2_name (str): The name of the second column for the operation.
            operation (callable): A function that takes two pandas Series (from col1 and col2)
                                  and returns a new pandas Series with the result.
                                  Example: lambda c1, c2: (c1 - c2) / (c1 + c2)
        """

        # Helper function to apply the operation to a dataframe
        def apply_op(df):
            if col1_name not in df.columns:
                raise ValueError(f"Column '{col1_name}' not found in the dataframe.")
            if col2_name not in df.columns:
                raise ValueError(f"Column '{col2_name}' not found in the dataframe.")

            # The provided operation function should handle potential issues like division by zero
            # by returning np.nan or similar, which is standard for pandas operations.
            df[new_col_name] = operation(df[col1_name], df[col2_name])
            return df

        # Apply the operation to raw_df
        self.raw_df = apply_op(self.raw_df.copy())

        # Apply the operation to filtered_df if it exists
        if self.filtered_df is not None:
            self.filtered_df = apply_op(self.filtered_df.copy())

        # Add the new column to the list of metric variables
        if new_col_name not in self.metric_variables:
            self.metric_variables.append(new_col_name)

    def scale_metrics(self, scaletype: str = "robust", per_group: bool = True):
        """
        Scales the metric variables in the dataset.

        This method applies scaling to the metric variables in both raw_df and filtered_df.
        The scaled data is stored in new columns with a '_scaled' suffix.

        Args:
            scaletype (str): The type of scaling to perform.
                             Options: 'standard' (StandardScaler), 'minmax' (MinMaxScaler),
                             'robust' (RobustScaler). Defaults to 'robust'.
            per_group (bool): If True and a grouping_variable is set, scaling is
                              performed within each group. Defaults to True.
        """

        scalers = {
            "standard": StandardScaler,
            "minmax": MinMaxScaler,
            "robust": RobustScaler,
        }
        if scaletype not in scalers:
            raise ValueError(f"Invalid scaletype. Choose from {list(scalers.keys())}")
        Scaler = scalers[scaletype]

        def apply_scaling(df):
            if df is None or df.empty:
                return df

            df_copy = df.copy()
            metrics_to_scale = self.metric_variables
            scaled_cols = [f"{col}_scaled" for col in metrics_to_scale]

            if per_group and self.grouping_variable:
                # Scale within each group
                grouped = df_copy.groupby(self.grouping_variable, group_keys=False)
                df_copy[scaled_cols] = grouped[metrics_to_scale].apply(
                    lambda x: Scaler().fit_transform(x) if not x.empty else x
                )
            else:
                # Scale across the entire dataframe
                df_copy[scaled_cols] = Scaler().fit_transform(df_copy[metrics_to_scale])
            return df_copy

        self.raw_df = apply_scaling(self.raw_df)
        if self.filtered_df is not None:
            self.filtered_df = apply_scaling(self.filtered_df)

    def calculate_si(self, new_col_name: str = "si_score", metrics_to_use: list = None):
        """
        Calculates social (or summary) index (SI) score by summing specified metric columns.

        Typically used on scaled metrics. The new SI score column is added to both
        raw_df and filtered_df.

        Args:
            new_col_name (str): The name for the new summary index column.
            metrics_to_use (list, optional): A list of metric column names to sum.
                                             If None, uses all columns in `self.metric_variables`
                                             that end with '_scaled'.
        """

        def apply_si(df):
            if df is None:
                return None

            df_copy = df.copy()
            if metrics_to_use is None:
                cols = [col for col in self.metric_variables if col.endswith("_scaled")]
                if not cols:
                    cols = [
                        col for col in df_copy.columns if col.endswith("_scaled") and col not in self.metric_variables
                    ]
            else:
                cols = metrics_to_use

            missing_cols = [col for col in cols if col not in df_copy.columns]
            if missing_cols:
                raise ValueError(f"Columns not found in dataframe: {missing_cols}")

            df_copy[new_col_name] = df_copy[cols].sum(axis=1)
            return df_copy

        self.raw_df = apply_si(self.raw_df)
        if self.filtered_df is not None:
            self.filtered_df = apply_si(self.filtered_df)

        if new_col_name not in self.metric_variables:
            self.metric_variables.append(new_col_name)

    def clean_missing_values(self, subset_cols: list = None):
        """
        Removes rows with missing values from the dataset.

        Operates on both raw_df and filtered_df.

        Args:
            subset_cols (list, optional): A list of column names to check for missing values.
                                          If None, all columns are checked.
        """

        def drop_na_rows(df):
            if df is None:
                return None
            return df.dropna(subset=subset_cols)

        self.raw_df = drop_na_rows(self.raw_df)
        if self.filtered_df is not None:
            self.filtered_df = drop_na_rows(self.filtered_df)
