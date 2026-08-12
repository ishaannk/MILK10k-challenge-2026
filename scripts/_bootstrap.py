"""Put the project root on ``sys.path`` so ``scripts/*.py`` can ``import src``.

Imported for its side effect at the top of every script. This keeps the scripts
runnable directly (``python scripts/train.py``) without requiring an editable
install or a ``PYTHONPATH`` export in every shell.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
