"""
reconcile_results.py
====================
Canonical top-level result reconciliation entrypoint.
Executes mathematical identity reconciliation with hard fail (exit code 1) on discrepancy.
"""

import sys
import subprocess

def main():
    cmd = [sys.executable, "scripts/reconcile_v2.py"] + sys.argv[1:]
    res = subprocess.run(cmd)
    sys.exit(res.returncode)

if __name__ == "__main__":
    main()
