"""Entry point for running the Streamlit frontend."""

import os
import sys

# Ensure the project root is on the path when running directly
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from frontend.streamlit_app import main

if __name__ == "__main__":
    main()
