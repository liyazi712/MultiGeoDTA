#!/usr/bin/env python3
"""Backward-compatible training entry (delegates to multigeodta CLI)."""
import sys

from multigeodta.cli import main

if __name__ == "__main__":
    sys.argv = ["multigeodta", "train"] + sys.argv[1:]
    raise SystemExit(main())
