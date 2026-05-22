"""Make the repo root importable so `import neuralflow` works under pytest
without an editable install."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
