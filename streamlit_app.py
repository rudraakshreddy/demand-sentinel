import sys
import runpy
from pathlib import Path

# Ensure project root is on sys.path for src.* imports
sys.path.insert(0, str(Path(__file__).parent))

if __name__ == "__main__":
    # Execute the dashboard script directly in the current namespace
    runpy.run_path("src/serving/dashboard.py", run_name="__main__")
