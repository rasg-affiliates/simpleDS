# -*- mode: python; coding: utf-8 -*-
# Copyright (c) 2020 rasg-affiliates
# Licensed under the 3-clause BSD License

"""Testing environment setup and teardown for pytest."""
import os
import shutil

import pytest
from simpleDS.data import DATA_PATH


@pytest.fixture(autouse=True, scope="session")
def setup_and_teardown_package():
    """Make data/test directory to put test output files in."""
    testdir = os.path.join(DATA_PATH, "test_data/")
    if not os.path.exists(testdir):
        print("making test directory")
        os.mkdir(testdir)

    yield

    shutil.rmtree(testdir)
