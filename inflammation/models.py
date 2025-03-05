"""Module containing models representing patients and their data.

The Model layer is responsible for the 'business logic' part of the software.

Patients' data is held in an inflammation table (2D array) where each row contains
inflammation data for a single patient taken over a number of days
and each column represents a single day across all patients.
"""

import os
import numpy as np

from typing import List, Dict, Any


def load_csv(filename):
    """Load a Numpy array from a CSV

    :param filename: Filename of CSV to load
    """
    return np.loadtxt(fname=filename, delimiter=",")


def daily_mean(data):
    """Calculate the daily mean of a 2d inflammation data array."""
    return np.mean(data, axis=0)


def daily_max(data):
    """Calculate the daily max of a 2d inflammation data array."""
    return np.max(data, axis=0)


def daily_min(data):
    """Calculate the daily min of a 2d inflammation data array."""
    return np.min(data, axis=0)


def patient_normalise(data):
    """
    Normalise patient data from a 2D inflammation data array.

    NaN values are ignored, and normalised to 0.

    Negative values are rounded to 0.
    """
    if np.any(data < 0):
        raise ValueError("Inflammation values should not be negative")

    max = np.nanmax(data, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        normalised = data / max[:, np.newaxis]
    normalised[np.isnan(normalised)] = 0
    normalised[normalised < 0] = 0
    return normalised


### decoumpling task ###
class Reader:
    """Creates an instance of the Reader class. This will take a list of file paths on initialization. Use the .read() method to read the files and return the data.

    Raises:
        ValueError: If the file type is not supported
        ValueError: If the file path is not valid

    Returns:
        Reader: Object instance.
    """

    inpaths: List[str]
    data_out: Dict[str, Any]

    def __init__(self, inpaths: List[str]) -> "Reader":
        self.inpaths = inpaths
        self.data_out = {}

    def read(self, return_data: bool = False, as_list: bool = False):
        for path in self.inpaths:
            match path.split(".")[-1]:
                case "csv":
                    self.data_out[path] = self.read_csv(path)
                case "txt":
                    self.data_out[path] = self.read_txt(path)
                case "npy":
                    self.data_out[path] = self.read_npy(path)
                case _:
                    raise ValueError(f"File type not supported: {path}")

        if as_list:
            return list(self.data_out.values())
        if not return_data:
            return
        return self.data_out

    def get_data(self):
        return self.data_out

    ## Helper
    def read_csv(self, path: str):
        return np.loadtxt(fname=path, delimiter=",")

    def read_txt(self, path: str):
        return np.loadtxt(fname=path)

    def read_npy(self, path: str):
        return np.load(path)

    ## checkups
    def check_files_provided(self, inpaths: List[str]):
        try:
            assert len(inpaths) > 0
            assert all([os.path.exists(path) for path in inpaths])
            assert all([os.path.isfile(path) for path in inpaths])

        except AssertionError:
            raise ValueError("Please provide a valid file path")
