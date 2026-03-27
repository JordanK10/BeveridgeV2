"""Shared default output directory for Beveridge figure exports."""

import os

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT_DIR = os.path.join(_PROJECT_ROOT, "output")
