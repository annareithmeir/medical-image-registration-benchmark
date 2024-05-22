import csv
import os
import statistics

import numpy as np


class EvaluationResults:
    def __init__(self, file_path):
        self.file_path = file_path

        # if the csv exists, delete it
        if os.path.exists(self.file_path):
            os.remove(self.file_path)

        open(self.file_path, 'w').close()

    def add_value(self, method: str, value, row_name: str):
        """
        Add a value to the specified method in the CSV file.

        If you want to add a new method you need to remove mean and std first

        Args:
            method (str): The method to add the value for.
            value: The value to add.

        Returns:
            None
        """

        # Read the current contents of the CSV file

        with open(self.file_path, mode='r', newline='') as file:
            reader = csv.reader(file)
            rows = list(reader)

        # Ensure there is a header row
        if len(rows) == 0:
            rows.append(['file'])

        # Find the column index for the given method, or the first empty column
        header = rows[0]
        column_index = None
        for idx, col_name in enumerate(header):
            if col_name == method:
                column_index = idx
                break
            elif col_name == '' and column_index is None:
                column_index = idx

        # If no empty column was found and the method is not in the header, add a column
        if column_index is None:
            column_index = len(header)
            header.append(method)

        # Update the header if necessary
        if header[column_index] == '':
            header[column_index] = method

        # Find the first empty row in the specified column
        row_index = None
        for idx, row in enumerate(rows[1:], start=1):
            if len(row) <= column_index or row[column_index] == '':
                row_index = idx
                break

        # If no empty row was found, add a new row
        if row_index is None:
            row_index = len(rows)
            rows.append([''])

        # Ensure the row has enough columns
        while len(rows[row_index]) <= column_index:
            rows[row_index].append('')

        # Add the value to the cell
        rows[row_index][column_index] = value

        rows[row_index][0] = row_name

        # Write the updated contents back to the CSV file
        with open(self.file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(rows)

    def calculate_min(self):
        self.__calculate_statistics(np.min, 'min')

    def calculate_max(self):
        self.__calculate_statistics(np.max, 'max')

    def calculate_mean(self):
        self.__calculate_statistics(statistics.mean, 'mean')

    def calculate_stddev(self):
        self.__calculate_statistics(statistics.stdev, 'std')

    def __calculate_statistics(self, stat_function: callable, function_name: str):
        # Read the current contents of the CSV file
        with open(self.file_path, mode='r', newline='') as file:
            reader = csv.reader(file)
            rows = list(reader)

        if len(rows) == 0:
            raise ValueError(
                f"The CSV file is empty. Cannot calculate {function_name}.")

        # Check if there is already a row with the word 'function_name'
        for row in rows:
            if function_name in row:
                print(f"{function_name} values already calculated.")
                return

        header = rows[0][1:]
        results = []

        # find the range of rows to calculate the values for (only where in the first column is a name)
        start = 1
        end = 1
        for idx, row in enumerate(rows[1:], start=1):
            if row[0] != '':
                end = idx
            else:
                break

        # Calculate the mean for each column
        for col_index in range(1, len(header) + 1):
            values = []

            for row_index in range(start, end + 1):
                if len(rows[row_index]) > col_index and rows[row_index][col_index] != '':
                    try:
                        values.append(float(rows[row_index][col_index]))
                    except ValueError:
                        continue
            if values:
                function_value = stat_function(values)
            else:
                function_value = None
            results.append(function_value)

        # Add rows for 'mean' and the calculated mean values
        result_row_label = [function_name if col !=
                            '' else '' for col in header]
        result_values_row = [
            str(result) if result is not None else '' for result in results]

        rows.append([''] + result_row_label)
        rows.append([''] + result_values_row)

        # Write the updated contents back to the CSV file
        with open(self.file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(rows)
