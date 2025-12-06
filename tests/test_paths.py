"""
Test environment-aware paths
"""
import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.paths import (
    PROJECT_ROOT,
    MODELS_DIR,
    DATA_DIR,
    MONITORING_DIR,
    get_project_root
)


def test_project_root_exists():
    """Test that project root is detected and exists"""
    assert PROJECT_ROOT is not None
    assert PROJECT_ROOT.exists()
    assert PROJECT_ROOT.is_dir()


def test_project_root_is_absolute():
    """Test that project root is an absolute path"""
    assert PROJECT_ROOT.is_absolute()


def test_key_directories_defined():
    """Test that key directories are defined"""
    assert MODELS_DIR is not None
    assert DATA_DIR is not None
    assert MONITORING_DIR is not None


def test_paths_are_absolute():
    """Test that all paths are absolute"""
    assert MODELS_DIR.is_absolute()
    assert DATA_DIR.is_absolute()
    assert MONITORING_DIR.is_absolute()


def test_get_project_root_function():
    """Test the get_project_root function works"""
    root = get_project_root()
    assert root is not None
    assert isinstance(root, Path)
    assert root.exists()


def test_directory_structure():
    """Test that expected directory structure exists or can be created"""
    expected_dirs = [
        PROJECT_ROOT / 'src',
        PROJECT_ROOT / 'data',
        PROJECT_ROOT / 'models',
        PROJECT_ROOT / 'monitoring'
    ]
    
    for dir_path in expected_dirs:
        assert dir_path.exists() or dir_path.parent.exists()