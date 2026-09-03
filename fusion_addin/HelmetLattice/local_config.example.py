"""Copy this file to `local_config.py` and edit the paths for your machine.

    cp local_config.example.py local_config.py

`local_config.py` is gitignored — your paths stay on your machine.
Anything you leave blank falls back to the HELMET_LATTICE_* environment
variables, then to the add-in's built-in defaults.
"""

# Absolute path to your clone of this repository.
REPO_DIR = r"C:\Users\YOU\helmet-lattice"

# A system Python (NOT Fusion's bundled one) that has the packages in
# requirements.txt installed. Find it with:  where python
PYTHON_EXE = r"C:\Users\YOU\AppData\Local\Programs\Python\Python313\python.exe"

# Master data table the optimizer learns from.
# Leave "" to use the pipeline default (data/raw/dummy_master.csv).
# Point at the real handoff CSV once testing has produced one.
DATA_CSV = ""
