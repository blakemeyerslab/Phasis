#!/usr/bin/env python3
"""Compatibility wrapper for the installed ``phasis-compare`` command."""

from phasis.result_comparison import main


if __name__ == "__main__":
    raise SystemExit(main())
