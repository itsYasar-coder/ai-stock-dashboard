"""Makes the project root importable when running bare `pytest` from anywhere."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
