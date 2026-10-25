#!/usr/bin/env python
"""
DevMoE - Single Command Entry Point
Usage:
    python run.py serve
    python run.py status
    python run.py recommend
    python run.py route "write a regex"
"""
import sys
from devmoe.cli import main

if __name__ == "__main__":
    # If run with no arguments, default to serve
    if len(sys.argv) == 1:
        sys.argv.append("serve")
    main()
