"""Tests for statistics functions within the Model layer."""

import numpy as np
import pytest

import numpy.testing as npt

from inflammation.models import (
    daily_mean,
    daily_min,
    daily_max,
    load_csv,
    patient_normalise,
)


def test_daily_mean_zeros():
    """Test that mean function works for an array of zeros."""

    test_input = np.array([[0, 0], [0, 0], [0, 0]])
    test_result = np.array([0, 0])

    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(daily_mean(test_input), test_result)


def test_daily_mean_integers():
    """Test that mean function works for an array of positive integers."""

    test_input = np.array([[1, 2], [3, 4], [5, 6]])
    test_result = np.array([3, 4])

    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(daily_mean(test_input), test_result)


def test_daily_min_integers():
    """Test that min function works for an array of positive integers."""

    test_input = np.array([[1, 2], [3, 4], [5, 6]])
    test_result = np.array([1, 2])

    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(daily_min(test_input), test_result)


def test_daily_max_integers():
    """Test that max function works for an array of positive integers."""

    test_input = np.array([[1, 2], [3, 4], [5, 6]])
    test_result = np.array([5, 6])

    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(daily_max(test_input), test_result)


def test_daily_min_string():
    """Test that min function works for an array of strings."""

    with pytest.raises(TypeError):
        daily_min(np.array([["Hello", "there"], ["General", "Kenobi"]]))


# %% Parametrized Tests
@pytest.mark.parametrize(
    "test, expected",
    [
        ([[0, 0], [0, 0], [0, 0]], [0, 0]),
        ([[1, 2], [3, 4], [5, 6]], [3, 4]),
    ],
)
def test_daily_mean(test, expected):
    """Test mean function works for array of zeroes and positive integers."""
    npt.assert_array_equal(daily_mean(np.array(test)), np.array(expected))


# %% Test for exceptions
def test_load_csv_missing_file():
    """Test that the load_csv function raises an exception for a missing file."""
    with pytest.raises(FileNotFoundError):
        load_csv("data/nonexistent-file.csv")


@pytest.mark.parametrize(
    "test, expected, expect_raises",
    [
        ([[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]], None),
        ([[1, 1, 1], [1, 1, 1], [1, 1, 1]], [[1, 1, 1], [1, 1, 1], [1, 1, 1]], None),
        (
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            [[0.33, 0.67, 1], [0.67, 0.83, 1], [0.78, 0.89, 1]],
            None,
        ),
        (
            [[-1, 2, 3], [4, 5, 6], [7, 8, 9]],
            [[0, 0.67, 1], [0.67, 0.83, 1], [0.78, 0.89, 1]],
            ValueError,
        ),
        (
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            [[0.33, 0.67, 1], [0.67, 0.83, 1], [0.78, 0.89, 1]],
            None,
        ),
    ],
)
def test_patient_normalise(test, expected, expect_raises):
    """Test normalisation works for arrays of one and positive integers."""

    if expect_raises is not None:
        with pytest.raises(expect_raises):
            result = patient_normalise(np.array(test))
            npt.assert_allclose(result, np.array(expected), rtol=1e-2, atol=1e-2)
    else:
        result = patient_normalise(np.array(test))
        npt.assert_allclose(result, np.array(expected), rtol=1e-2, atol=1e-2)


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
