#!/usr/bin/env python3
"""
Backward-compatible JEPA pretraining entrypoint.

Use `scripts/train_jepa_pretrain.py` for new runs.
"""

from __future__ import annotations

from train_jepa_pretrain import *  # noqa: F401,F403
from train_jepa_pretrain import main


if __name__ == "__main__":
    print(
        "[DEPRECATED] scripts/train_step1_pretrain.py is kept for compatibility. "
        "Please use scripts/train_jepa_pretrain.py"
    )
    main()
