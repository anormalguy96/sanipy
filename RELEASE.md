# Release Checklist

Use this checklist to prepare, verify, and publish a new release of **Sanipy**.

## Phase 1: Pre-Release Verification

1. **Clean Working Tree**:
   - Ensure all changes are committed and the git status is clean:
     ```bash
     git status
     ```

2. **Run All Tests**:
   - Run the complete unit and integration test suite:
     ```bash
     pytest -v
     ```

3. **Changelog Consistency Check**:
   - Verify `CHANGELOG.md` is updated with the new version section, date, and listed improvements.

4. **Version Consistency Check**:
   - Verify the version string in `pyproject.toml` (e.g. `version = "0.1.0a4"`) matches `__version__` in `src/sanipy/__init__.py`.
   - Run the version validation test:
     ```bash
     pytest tests/test_integration.py -k test_version_consistency
     ```

---

## Phase 2: Build & Package Validation

1. **Clean Stale Build Artifacts**:
   - Remove previous build directories and packages:
     - On Windows (PowerShell):
       ```powershell
       Remove-Item -Recurse -Force dist, build, *.egg-info -ErrorAction SilentlyContinue
       ```
     - On Linux/macOS:
       ```bash
       rm -rf dist build *.egg-info
       ```

2. **Build Distribution Packages**:
   - Build the source distribution and binary wheel:
     ```bash
     python -m build
     ```

3. **Validate Metadata with Twine**:
   - Verify that description rendering and metadata are 100% correct:
     ```bash
     twine check dist/*
     ```

---

## Phase 3: Local Installation Smoke Test

1. **Create Isolated Environment**:
   - Set up a clean virtual environment and upgrade pip:
     - On Windows (PowerShell):
       ```powershell
       py -m venv test_env
       test_env\Scripts\python -m pip install --upgrade pip
       test_env\Scripts\python -m pip install dist/*.whl
       ```
     - On Linux/macOS:
       ```bash
       python -m venv test_env
       test_env/bin/python -m pip install --upgrade pip
       test_env/bin/python -m pip install dist/*.whl
       ```

2. **Verify CLI Executable & Functionality**:
   - Confirm version flags:
     - Windows: `test_env\Scripts\sanipy --version`
     - Linux/macOS: `test_env/bin/sanipy --version`
   - Run sanity checks:
     - Windows: `test_env\Scripts\sanipy check examples/messy_classification.csv --target churn`
     - Linux/macOS: `test_env/bin/sanipy check examples/messy_classification.csv --target churn`
   - Run train/test comparison checks:
     - Windows: `test_env\Scripts\sanipy compare examples/train_classification.csv examples/test_classification.csv --target churn --task classification`
     - Linux/macOS: `test_env/bin/sanipy compare examples/train_classification.csv examples/test_classification.csv --target churn --task classification`

3. **Clean Up Environment**:
   - Delete the temporary `test_env/` directory when done.

---

## Phase 4: Staging & Publishing

1. **Publish to TestPyPI**:
   - Upload the package to the TestPyPI registry:
     ```bash
     python -m twine upload --repository testpypi dist/*
     ```
     *(Use token authentication configured via a secure `~/.pypirc` or input the API token when prompted. Do NOT hardcode or commit API tokens to git. If a token is ever accidentally shared or exposed, revoke it immediately via your PyPI/TestPyPI dashboard.)*

2. **Verify Staging Installation**:
   - Create a separate clean virtual environment and install the package from TestPyPI:
     - On Windows (PowerShell):
       ```powershell
       py -m venv test_pypi_env
       test_pypi_env\Scripts\python -m pip install --upgrade pip
       test_pypi_env\Scripts\python -m pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple sanipy
       ```
     - On Linux/macOS:
       ```bash
       python -m venv test_pypi_env
       test_pypi_env/bin/python -m pip install --upgrade pip
       test_pypi_env/bin/python -m pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple sanipy
       ```
     *(Note: `--extra-index-url` ensures required standard dependencies like `pandas` and `numpy` can be fetched from real PyPI if unavailable on TestPyPI).*

3. **Run CLI Smoke Tests from TestPyPI**:
   - Verify that the command `sanipy --version` and check/compare tools work identically inside the `test_pypi_env` environment.
   - Delete `test_pypi_env/` when complete.

4. **Publish to Real PyPI**:
   - Upload distribution files only after all prior checks pass flawlessly:
     ```bash
     python -m twine upload dist/*
     ```

5. **Tag Release in Git**:
   - Create and push a signed/annotated git tag matching the released version:
     ```bash
     git tag -a v0.1.0a4 -m "Release v0.1.0a4"
     git push origin v0.1.0a4
     ```
