"""Run Sanipy as a module.

This enables:

    python -m sanipy
    python -m sanipy --help
    python -m sanipy --version
    python -m sanipy check ...
    python -m sanipy compare ...
"""

from sanipy.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
