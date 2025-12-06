"""
Pytest configuration and fixtures
"""
import pytest
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture(scope="session")
def project_root_path():
    """Provide project root path to tests"""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def test_data_dir(project_root_path):
    """Provide test data directory"""
    test_dir = project_root_path / "tests" / "test_data"
    test_dir.mkdir(exist_ok=True)
    return test_dir