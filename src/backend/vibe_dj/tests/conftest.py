"""Shared pytest configuration for vibe_dj tests."""

import sys
from pathlib import Path

# Ensure vibe_dj package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
