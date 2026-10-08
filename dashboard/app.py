"""
dashboard/app.py
================
Streamlit Security Operations Center & Analytics Entrypoint.
"""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Execute main streamlit application
import app.streamlit_app
