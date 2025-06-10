import sys
import os
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '')))

@pytest.fixture
def sample_fixture():
    return "sample data"