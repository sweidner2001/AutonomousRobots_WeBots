"""
Controller_v1.py  --  Webots entry point for the RosBot Graph-SLAM mapper.
==========================================================================
THIS IS THE FILE WEBOTS LAUNCHES.

In the Webots scene tree the Rosbot node's ``controller`` field is set to
"Controller_v1", so Webots runs this file.  It is intentionally tiny: it only
fixes up the import path and hands control to ``SlamApp``.  All the real work
lives in well-named classes (see app.py and the slam/ package).

IMPORT PATH
-----------
The modules import each other as ``Maze5.controllers.Controller_v1.*``, so the
folder that CONTAINS the ``Maze5`` package (the repo root, ``sys-code``) must
be on ``sys.path``.  This file is at

    .../sys-code/Maze5/controllers/Controller_v1/Controller_v1.py

so the repo root is three directories up from this file's folder.
"""

import os
import sys

# Add the repo root (the folder containing the "Maze5" package) to sys.path.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from Maze5.controllers.Controller_v1.app import SlamApp


def main():
    SlamApp().run()


if __name__ == "__main__":
    main()
