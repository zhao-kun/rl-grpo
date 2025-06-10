# This file can remain empty to mark the tests directory as a package.
import os
import sys

path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(f"path for src: {path}")
sys.path.append(path)