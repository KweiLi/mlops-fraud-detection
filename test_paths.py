"""
Test script to verify paths work in both local and Docker environments
"""
import sys
from pathlib import Path

print("="*70)
print("ENVIRONMENT-AWARE PATHS TEST")
print("="*70)

# Test 1: Import paths utility
print("\n[TEST 1] Importing paths utility...")
try:
    from src.utils.paths import (
        PROJECT_ROOT,
        MODELS_DIR,
        DATA_DIR,
        MONITORING_DIR,
        FEATURE_NAMES_FILE,
        TRANSACTIONS_FILE,
        get_latest_model,
    )
    print("✓ Import successful")
except Exception as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Check project root
print(f"\n[TEST 2] Checking project root...")
print(f"  Path: {PROJECT_ROOT}")
print(f"  Exists: {PROJECT_ROOT.exists()}")
print(f"  Is absolute: {PROJECT_ROOT.is_absolute()}")
if PROJECT_ROOT.exists():
    print("✓ Project root OK")
else:
    print("✗ Project root not found")
    sys.exit(1)

# Test 3: Check directories
print(f"\n[TEST 3] Checking directories...")
directories = {
    'models': MODELS_DIR,
    'data': DATA_DIR,
    'monitoring': MONITORING_DIR,
}

all_dirs_ok = True
for name, path in directories.items():
    exists = path.exists()
    print(f"  {name}: {path} {'✓' if exists else '✗'}")
    if not exists:
        all_dirs_ok = False

if all_dirs_ok:
    print("✓ All directories found")
else:
    print("✗ Some directories missing")

# Test 4: Check files
print(f"\n[TEST 4] Checking files...")
files = {
    'feature_names': FEATURE_NAMES_FILE,
    'transactions': TRANSACTIONS_FILE,
}

all_files_ok = True
for name, path in files.items():
    exists = path.exists()
    size = path.stat().st_size if exists else 0
    print(f"  {name}: {path}")
    print(f"    Exists: {exists} {'✓' if exists else '✗'}")
    if exists:
        print(f"    Size: {size:,} bytes")
    if not exists:
        all_files_ok = False

if all_files_ok:
    print("✓ All files found")
else:
    print("✗ Some files missing")

# Test 5: Get latest model
print(f"\n[TEST 5] Getting latest model...")
try:
    latest_model = get_latest_model()
    print(f"  Path: {latest_model}")
    print(f"  Name: {latest_model.name}")
    print(f"  Size: {latest_model.stat().st_size / (1024*1024):.2f} MB")
    print("✓ Latest model found")
except Exception as e:
    print(f"✗ Failed: {e}")

# Test 6: Try loading model (if available)
print(f"\n[TEST 6] Testing model loading...")
try:
    import joblib
    latest_model = get_latest_model()
    model = joblib.load(latest_model)
    print(f"  Model type: {type(model).__name__}")
    print("✓ Model loaded successfully")
except Exception as e:
    print(f"✗ Failed: {e}")

# Test 7: Environment detection
print(f"\n[TEST 7] Environment detection...")
if Path('/app').exists():
    print("  Environment: Docker 🐳")
    print(f"  Working directory: {Path.cwd()}")
else:
    print("  Environment: Local 💻")
    print(f"  Working directory: {Path.cwd()}")

# Summary
print("\n" + "="*70)
print("TEST SUMMARY")
print("="*70)
print(f"Project Root: {PROJECT_ROOT}")
print(f"Environment: {'Docker' if Path('/app').exists() else 'Local'}")
print(f"All Critical Paths: {'✓ WORKING' if all_dirs_ok and all_files_ok else '✗ ISSUES'}")
print("="*70)