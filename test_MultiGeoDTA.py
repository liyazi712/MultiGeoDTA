#!/usr/bin/env python3
"""Backward-compatible evaluation entry (delegates to multigeodta CLI)."""
import sys

from multigeodta.cli import main

if __name__ == "__main__":
    sys.argv = ["multigeodta", "evaluate"] + sys.argv[1:]
    raise SystemExit(main())
