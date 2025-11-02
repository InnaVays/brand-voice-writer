import sys, os

def add_repo_root():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.normpath(os.path.join(here, ".."))
    if root not in sys.path:
        sys.path.insert(0, root)
