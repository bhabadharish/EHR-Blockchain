"""
generate_results.py
===================
Canonical top-level result generation script.
Regenerates all benchmark CSVs, JSONs, and paper tables.
"""

import sys
import subprocess

def main():
    cmd = [sys.executable, "scripts/generate_all_v2_results.py"] + sys.argv[1:]
    res = subprocess.run(cmd)
    sys.exit(res.returncode)

if __name__ == "__main__":
    main()
