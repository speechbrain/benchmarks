"""Tests for checking the docstrings of functions and classes.

Authors
 * Mirco Ravanelli 2022
 * Bruno Aristimunha 2025
"""
import os
from tests.utils.check_docstrings import check_docstrings


def test_recipe_list():
    base_folder = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..")
    )
    check_folders = ["speechbrain", "tools", "benchmarks/MOABB"]
    assert check_docstrings(base_folder, check_folders)
