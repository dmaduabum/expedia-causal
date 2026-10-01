"""
src/pipeline/extract.py
-----------------------
Extracts train.csv from the nested Kaggle zip using system unzip.
"""

import os
import subprocess

ROOT     = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DIR  = os.path.join(ROOT, "data", "raw")
ZIP_PATH = os.path.join(RAW_DIR, "expedia-personalized-sort.zip")
OUT_PATH = os.path.join(RAW_DIR, "train.csv")
TMP_DIR  = os.path.join(RAW_DIR, "tmp_extract")

def extract():
    if not os.path.exists(ZIP_PATH):
        raise FileNotFoundError(
            f"Raw zip not found at:\n  {ZIP_PATH}\n"
            "Download from: https://www.kaggle.com/c/expedia-personalized-sort/data"
        )

    if os.path.exists(OUT_PATH):
        print(f"train.csv already exists — skipping extraction.")
        return

    os.makedirs(TMP_DIR, exist_ok=True)

    # Step 1: unzip outer zip → gets data.zip into TMP_DIR
    print("Extracting outer zip...")
    subprocess.run(["unzip", "-o", ZIP_PATH, "data.zip", "-d", TMP_DIR], check=True)

    # Step 2: unzip inner data.zip → gets train.csv
    inner_zip = os.path.join(TMP_DIR, "data.zip")
    print("Extracting train.csv from data.zip...")
    subprocess.run(["unzip", "-o", inner_zip, "train.csv", "-d", RAW_DIR], check=True)

    # Step 3: cleanup tmp
    import shutil
    shutil.rmtree(TMP_DIR)

    print(f"Done. train.csv saved to: {OUT_PATH}")

if __name__ == "__main__":
    extract()
