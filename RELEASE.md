# Release Checklist

Use this checklist to prepare and publish a new release of **Sanipy**.

## Prep

1. **Tests & Style Verification**:
   - Ensure all tests pass:
     ```bash
     pytest
     ```

2. **Changelog**:
   - Update `CHANGELOG.md` with the new version, date, and changes.

3. **Version Check**:
   - Update the version string in `pyproject.toml` (e.g., `version = "0.1.0a1"`).

## Building Distributions

1. Build source archive and wheel:
   ```bash
   python -m build
   ```
2. Verify package descriptions are formatted correctly:
   ```bash
   twine check dist/*
   ```

## Staging & Publishing

1. **Publish to TestPyPI** (optional, recommended for major changes):
   ```bash
   twine upload --repository testpypi dist/*
   ```
   Verify installation in a clean environment:
   ```bash
   pip install --index-url https://test.pypi.org/simple/ --no-deps sanipy
   ```

2. **Publish to PyPI**:
   ```bash
   twine upload dist/*
   ```

3. **Tag Release**:
   - Create a Git tag and push it:
     ```bash
     git tag -a v0.1.0a1 -m "Release v0.1.0a1"
     git push origin v0.1.0a1
     ```
