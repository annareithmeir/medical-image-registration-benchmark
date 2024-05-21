
import os
import unittest
from pathlib import Path
import sys
from time import sleep

sys.path.append(str(Path(__file__).parent.absolute().parent.parent))  # nopep8

from registrationbaselines.core import result_csv


class TestEvaluationResults(unittest.TestCase):

    def test_create_csv_file_on_init(self):
        path = "temp.csv"

        _ = result_csv.EvaluationResults(path)

        self.assertTrue(os.path.exists(path))

        # if the csv exists, delete it
        if os.path.exists(path):
            os.remove(path)

    def test_overwrite_csv_on_init(self):
        # create a file and then check if it is overwritten
        path = "temp.csv"
        open(path, 'w').close()

        # get creation time of the file
        creation_time = os.path.getctime(path)

        sleep(0.01)

        _ = result_csv.EvaluationResults(path)

        # get creation time of the file after the class is created
        creation_time_after = os.path.getctime(path)

        self.assertNotEqual(creation_time, creation_time_after)

        # if the csv exists, delete it
        if os.path.exists(path):
            os.remove(path)

    def test_add_value(self):
        path_before = "registrationbaselines/tests/test_files/temp.csv"
        path_after = "registrationbaselines/tests/test_files/results_add_value_1.csv"

        res = result_csv.EvaluationResults(path_before)

        res.add_value("dice", 0.856174669362918, "LungCT_0001_0000")
        res.add_value("hausdorff", 41.6293165929973, "LungCT_0001_0000")
        res.add_value("sdlogj", 0.171325790782495, "LungCT_0001_0000")
        res.add_value("num_foldings", 75073, "LungCT_0001_0000")

        res.add_value("dice", 0.961881779472523, "LungCT_0002_0000")
        res.add_value("hausdorff", 17, "LungCT_0002_0000")
        res.add_value("sdlogj", 0.123892355886574, "LungCT_0002_0000")
        res.add_value("num_foldings", 142862, "LungCT_0002_0000")

        # check if both files match
        with open(path_before, 'r') as f1, open(path_after, 'r') as f2:
            self.assertEqual(f1.read(), f2.read())

        # if the csv exists, delete it
        if os.path.exists(path_before):
            os.remove(path_before)


if __name__ == '__main__':
    unittest.main()
