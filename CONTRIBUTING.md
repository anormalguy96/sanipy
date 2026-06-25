# Contributing to Sanipy

Thank you for your interest in contributing to **Sanipy**! We welcome bug reports, feature requests, and code contributions.

## Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/anormalguy96/sanipy.git
   cd sanipy
   ```

2. Install the package in editable mode with development dependencies:
   ```bash
   python -m pip install -e ".[dev]"
   ```

## Running Tests

We use `pytest` for unit and integration testing. Run the following command from the project root:
```bash
pytest
```

To run tests with code coverage:
```bash
pytest --cov=sanipy
```

## Creating a Pull Request

1. Fork the repository and create your branch from `main`.
2. Write tests that cover your changes.
3. Verify that all tests pass locally.
4. Keep the library footprint minimal: do not add heavy dependencies.
5. Follow PEP 8 guidelines and include type hints for new functions.
6. Open a Pull Request detailing your changes.
