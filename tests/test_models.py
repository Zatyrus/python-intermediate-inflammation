"""Tests for statistics functions within the Model layer."""

import numpy as np
import numpy.testing as npt


def test_daily_mean_zeros():
    """Test that mean function works for an array of zeros."""
    from inflammation.models import daily_mean

    test_input = np.array([[0, 0], [0, 0], [0, 0]])
    test_result = np.array([0, 0])

    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(daily_mean(test_input), test_result)


def test_daily_mean_integers():
    """Test that mean function works for an array of positive integers."""
    from inflammation.models import daily_mean

    test_input = np.array([[1, 2], [3, 4], [5, 6]])
    test_result = np.array([3, 4])

    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(daily_mean(test_input), test_result)


def test_reader():
    """Test that Reader class can read in
    multiple files and return the data."""
    from inflammation.models import Reader

    test_input = ["data/inflammation-01.csv", "data/inflammation-02.csv"]
    test_result = {
        "data/inflammation-01.csv": np.loadtxt(
            "data/inflammation-01.csv", delimiter=","
        ),
        "data/inflammation-02.csv": np.loadtxt(
            "data/inflammation-02.csv", delimiter=","
        ),
    }

    reader = Reader(test_input)
    reader.read()

    for key, value in reader.data_out.items():
        npt.assert_array_equal(value, test_result[key])

    for key, known_key in zip(reader.data_out.keys(), test_result.keys()):
        npt.assert_string_equal(key, known_key)
