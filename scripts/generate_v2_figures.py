#!/usr/bin/env python3
"""scripts/generate_v2_figures.py
==============================
Publication Figures Generator for HAB-IDS & Blockchain Benchmarks.
Invokes the master generator scripts/generate_all_figures_hab_ids.py.
"""

import os
import sys

# Insert project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scripts.generate_all_figures_hab_ids import main as generate_all_figures


def main():
    generate_all_figures()


if __name__ == "__main__":
    main()
