import os

import numpy as np
import pandas as pd


class EvaluationResults:
    """
    A class to handle evaluation results and store them in a DataFrame.
    """

    def __init__(self, file_path: str):
        """
        Initializes the EvaluationResults class.

        Args:
            file_path (str): The path to the CSV file to save the results.
        """
        self.file_path = file_path
        self.df = pd.DataFrame()

        # If the CSV exists, delete it
        if os.path.exists(self.file_path):
            os.remove(self.file_path)

        open(self.file_path, 'w').close()

    def add_value(self, method: str, value, row_name: str):
        """
        Add a value to the specified method in the DataFrame.

        Args:
            method (str): The method to add the value for.
            value: The value to add.
            row_name (str): The row name for the value.

        Returns:
            None
        """
        if row_name not in self.df.index:
            self.df.loc[row_name, method] = value
        else:
            self.df.at[row_name, method] = value

    def calculate_min(self):
        """
        Calculate the minimum values for each method and add them as a row in the DataFrame.

        Returns:
            None
        """
        self.__calculate_statistics(np.min, 'min')

    def calculate_max(self):
        """
        Calculate the maximum values for each method and add them as a row in the DataFrame.

        Returns:
            None
        """
        self.__calculate_statistics(np.max, 'max')

    def calculate_mean(self):
        """
        Calculate the mean values for each method and add them as a row in the DataFrame.

        Returns:
            None
        """
        self.__calculate_statistics(np.mean, 'mean')

    def calculate_stddev(self):
        """
        Calculate the standard deviation values for each method and add them as a row in the DataFrame.

        Returns:
            None
        """
        self.__calculate_statistics(np.std, 'std')

    def __calculate_statistics(self, stat_function: callable, function_name: str):
        """
        Calculate a specified statistic for each method and add it as a row in the DataFrame.

        Args:
            stat_function (callable): The function to use for calculating the statistic.
            function_name (str): The name of the statistic (e.g., 'min', 'max', 'mean', 'std').

        Returns:
            None

        Raises:
            ValueError: If the DataFrame is empty.
        """
        if self.df.empty:
            raise ValueError(
                f"The DataFrame is empty. Cannot calculate {function_name}.")

        if function_name in self.df.index:
            print(f"{function_name} values already calculated.")
            return

        stats = self.df.apply(stat_function)
        self.df.loc[function_name] = stats

    def write(self):
        """
        Write the DataFrame to a CSV file at the specified path.

        Returns:
            None
        """
        # sort alphabetically
        self.df = self.df.sort_index(axis=1)

        # rounding to 6 significant digits
        for col in self.df.columns:
            if self.df[col].dtype == float:
                self.df[col] = self.df[col].apply(lambda x: f"{x:.6g}")

        self.df.to_csv(self.file_path)
