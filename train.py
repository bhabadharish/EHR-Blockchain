"""
train.py
========
Canonical top-level training entry point for CA-HTDNet V2.
"""

import sys
import subprocess

def main():
    cmd = [sys.executable, "scripts/train_ca_htdnet_v2.py"] + sys.argv[1:]
    res = subprocess.run(cmd)
    sys.exit(res.returncode)

if __name__ == "__main__":
    main()
