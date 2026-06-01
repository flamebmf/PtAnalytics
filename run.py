# Copyright (c) 2026 PlurumTech.com
# SPDX-License-Identifier: GPL-3.0-only
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.main import run

if __name__ == "__main__":
    run()
