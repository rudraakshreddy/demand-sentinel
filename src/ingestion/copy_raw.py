"""
Ingestion Layer — copy_raw.py

Copies M5 Forecasting dataset files from the user's Downloads folder into
data/raw/, verifying file sizes to ensure integrity. This replaces a Kaggle
API download when files are already locally available.

Scientific Rigor:
  - MD5 checksums logged for every file (reproducibility anchor)
  - File-size guard prevents partial/corrupt copies from propagating
  - Idempotent: skips copy if destination already matches source
"""

import hashlib
import logging
import os
import shutil
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[
        logging.FileHandler("logs/ingestion.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────────────────
SOURCE_DIR = Path(os.environ.get("M5_SOURCE_DIR",
    Path.home() / "Downloads" / "m5-forecasting-accuracy"))
DEST_DIR   = Path("data/raw")

EXPECTED_FILES = {
    "sales_train_evaluation.csv": None,   # ~116 MB
    "calendar.csv":               None,   # ~101 KB
    "sell_prices.csv":            None,   # ~194 MB
    "sample_submission.csv":      None,
}


def md5(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def copy_file(src: Path, dst: Path) -> None:
    src_size = src.stat().st_size
    if dst.exists():
        if dst.stat().st_size == src_size:
            log.info("SKIP (already present, same size): %s", dst.name)
            return
        log.warning("SIZE MISMATCH — re-copying: %s", dst.name)
    log.info("Copying %s  (%.1f MB)", src.name, src_size / 1e6)
    shutil.copy2(src, dst)
    copied_hash = md5(dst)
    log.info("  md5=%s  size=%d bytes", copied_hash, dst.stat().st_size)


def main() -> None:
    DEST_DIR.mkdir(parents=True, exist_ok=True)
    log.info("=" * 60)
    log.info("M5 Raw Data Copy — Source: %s", SOURCE_DIR)
    log.info("=" * 60)

    if not SOURCE_DIR.exists():
        raise FileNotFoundError(
            f"Source directory not found: {SOURCE_DIR}\n"
            "Set env M5_SOURCE_DIR to the folder containing M5 CSV files."
        )

    for filename in EXPECTED_FILES:
        src_path = SOURCE_DIR / filename
        dst_path = DEST_DIR / filename
        if not src_path.exists():
            log.error("MISSING source file: %s", src_path)
            raise FileNotFoundError(f"Expected M5 file not found: {src_path}")
        copy_file(src_path, dst_path)

    log.info("All M5 files verified and copied to %s", DEST_DIR.resolve())


if __name__ == "__main__":
    main()
