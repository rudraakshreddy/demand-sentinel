import sys
from pathlib import Path

# Ensure project root is on sys.path for src.* imports
sys.path.insert(0, str(Path(__file__).parent))

from src.serving.dashboard import main

if __name__ == "__main__":
    main()
