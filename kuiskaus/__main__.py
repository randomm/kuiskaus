"""
Main entry point for Kuiskaus when run as a module
"""

import sys

if len(sys.argv) > 1 and sys.argv[1] == "--cli":
    from .app import main
else:
    from .menubar import main

if __name__ == "__main__":
    main()