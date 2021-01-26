# -*- mode: python; coding: utf-8 -*-
# Copyright (c) 2020 rasg-affiliates
# Licensed under the 3-clause BSD License

"""Testing environment setup and teardown for pytest."""
import pytest


@pytest.fixture(autouse=True, scope="session")
def setup_and_teardown_package():
    """Make data/test directory to put test output files in."""
    yield
