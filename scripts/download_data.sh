#!/usr/bin/env bash
# Fetch the MILK10k challenge data into data/raw/ (~345 MB compressed).
# Data is CC-BY-NC 4.0 and is not redistributed in this repository.
#
#   bash scripts/download_data.sh
set -euo pipefail

cd "$(dirname "$0")/.."
BASE=https://isic-archive.s3.amazonaws.com/challenges/milk10k
DEST=data/raw
mkdir -p "$DEST"

for f in \
  MILK10k_Training_Metadata.csv \
  MILK10k_Training_Supplement.csv \
  MILK10k_Training_GroundTruth.csv \
  MILK10k_Test_Metadata.csv \
  MILK10k_Test_Input.zip \
  MILK10k_Training_Input.zip
do
  if [[ -f "$DEST/$f" ]]; then
    echo "have $f"
  else
    echo "fetching $f"
    curl -fL --retry 5 --retry-delay 3 -o "$DEST/$f" "$BASE/$f"
  fi
done

# Python's zipfile is used rather than `unzip`, which is absent from many slim images.
python - <<'PY'
import zipfile, pathlib
dest = pathlib.Path("data/raw")
for name in ("MILK10k_Training_Input", "MILK10k_Test_Input"):
    if (dest / name).is_dir():
        print(f"have {name}/")
        continue
    print(f"extracting {name}.zip")
    with zipfile.ZipFile(dest / f"{name}.zip") as z:
        z.extractall(dest)
PY

echo
echo "Layout: data/raw/MILK10k_{Training,Test}_Input/<lesion_id>/<isic_id>.jpg"
echo "Next:   python scripts/prepare_data.py --config configs/base.yaml --check-images"
echo
echo "Tip: if data/ is on a network mount, copy the image dirs to local disk and point"
echo "     --set data.train_image_root=... at them (measured 2.5x faster per worker)."
