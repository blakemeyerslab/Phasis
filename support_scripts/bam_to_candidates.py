#!/usr/bin/env python3
"""Run after installing Phasis and the optional pysam dependency.

Usage and alignment contract: docs/external_bam.md.
"""
from phasis.external_bam import main

if __name__ == '__main__':
    raise SystemExit(main())
