"""Download SoccerNet-GSR splits from Hugging Face and unzip them in TrackLab's layout.

Usage:  python tools/get_gsr.py <data_dir> train valid
Result: <data_dir>/SoccerNetGS/<split>/SNGS-xxx/...  (zips are deleted after extraction)

The dataset moved to Hugging Face (SoccerNet/SN-GSR-2025); TrackLab's built-in
download still points at the old, removed link, so use this script instead.
Do not download 'test' on shared machines: test is used once, at the end (Team rules).
"""
import os
import sys
import zipfile

from huggingface_hub import hf_hub_download

ALLOWED = {"train", "valid", "test", "challenge"}


def main():
    if len(sys.argv) < 3 or not set(sys.argv[2:]) <= ALLOWED:
        sys.exit(__doc__)
    root = os.path.join(os.path.abspath(sys.argv[1]), "SoccerNetGS")
    zdir = os.path.join(root, "gamestate-2025")
    for split in sys.argv[2:]:
        print(f"== downloading {split}", flush=True)
        path = hf_hub_download("SoccerNet/SN-GSR-2025", f"{split}.zip",
                               repo_type="dataset", local_dir=zdir)
        print(f"== extracting {split}", flush=True)
        with zipfile.ZipFile(path) as zf:
            zf.extractall(os.path.join(root, split))
        os.remove(path)
        n = len([d for d in os.listdir(os.path.join(root, split)) if d.startswith("SNGS-")])
        print(f"== done {split}: {n} clips", flush=True)


if __name__ == "__main__":
    main()
