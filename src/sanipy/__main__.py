"""Module entrypoint for sanipy. Allows execution via python -m sanipy."""

from __future__ import annotations

import sys
from sanipy.cli import main

if __name__ == "__main__":
    sys.exit(main())
