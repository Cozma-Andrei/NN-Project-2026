#!/bin/bash
# Download SoccerNet-GSR (train + valid only) and build the training packs, then check that they match
# the team's packs exactly (same clip, frame and crop counts; same Task D labels.csv checksum).
#
#   bash tools/setup_data.sh D        # only pack D (jersey)
#   bash tools/setup_data.sh A B D    # several packs
#   bash tools/setup_data.sh none     # only download + check the full dataset (Task C, pipeline runs)
#   PACK_ONLY=1 bash tools/setup_data.sh D   # you received the pack as an archive: skip the dataset, only check the pack
#
# Run from the sn-gamestate repository root with `conda activate tracklab`.
# Needs ~21 GB for the download (train 9.8 GB + valid 11.2 GB zips, ~23 GB unpacked) plus the packs
# (A 1.8 GB, B 1.1 GB, D 0.9 GB). The zips are deleted after unpacking.
# 'test' is never downloaded here (Team rules: used once, at the end, by the integration owner).
set -euo pipefail

[ -f tools/get_gsr.py ] && [ -f tools/pack_gsr.py ] || { echo "run this from the sn-gamestate repo root, with tools/ copied in"; exit 1; }
python -c "import huggingface_hub, PIL" 2>/dev/null || { echo "activate the env first: conda activate tracklab"; exit 1; }
[ $# -ge 1 ] || { sed -n 2,12p "$0"; exit 1; }

DATA=data
GS=$DATA/SoccerNetGS
fail=0
check() { if [ "$2" == "$3" ]; then echo "  OK   $1: $2"; else echo "  FAIL $1: got $2, expected $3"; fail=1; fi; }

# 1. full dataset: download only the splits that are missing
if [ "${PACK_ONLY:-0}" == "1" ]; then
  echo "== PACK_ONLY=1: skipping the full dataset"
else
for split in train valid; do
  if [ -d "$GS/$split" ] && [ "$(ls "$GS/$split" | grep -c '^SNGS-')" -gt 0 ]; then
    echo "== $split already present, not downloading again"
  else
    python tools/get_gsr.py "$DATA" "$split"
  fi
done
echo "== checking the full dataset"
check "train clips" "$(ls $GS/train | grep -c '^SNGS-')" 57
check "valid clips" "$(ls $GS/valid | grep -c '^SNGS-')" 58
fi

# 2. packs
declare -A EXPECT=(
  [A]='{"train": {"clips": 57, "frames": 4275}, "valid": {"clips": 58, "frames": 4350}}'
  [B]='{"train": {"clips": 57, "frames": 4275}, "valid": {"clips": 58, "frames": 4350}}'
  [D]='{"train": {"clips": 57, "crops": 125677, "tracks": 1224}, "valid": {"clips": 58, "crops": 64518, "tracks": 1220}}'
)
for t in "$@"; do
  [ "$t" == "none" ] && continue
  [ -n "${EXPECT[$t]:-}" ] || { echo "unknown pack '$t' (use A, B or D; Task C uses the full dataset)"; exit 1; }
  if [ -f "$DATA/packs/$t/MANIFEST.json" ]; then
    echo "== pack $t already built, only checking it"
  elif [ "${PACK_ONLY:-0}" == "1" ]; then
    echo "PACK_ONLY=1 but $DATA/packs/$t/MANIFEST.json is missing: unpack the archive into $DATA/packs/$t/ first"; exit 1
  else
    echo "== building pack $t"
    python tools/pack_gsr.py --task "$t"
  fi
  echo "== checking pack $t"
  got=$(python -c "import json; print(json.dumps(json.load(open('$DATA/packs/$t/MANIFEST.json'))['counts'], sort_keys=True))")
  want=$(python -c "import json; print(json.dumps(json.loads('${EXPECT[$t]}'), sort_keys=True))")
  check "pack $t counts" "$got" "$want"
  if [ "$t" == "D" ]; then
    check "pack D labels.csv md5" "$(md5sum $DATA/packs/D/labels.csv | cut -d' ' -f1)" c317ff3eb074ab87dd8ed42b0219d27a
  fi
done

if [ $fail -eq 0 ]; then echo "ALL CHECKS PASSED: your data matches the team's."; else echo "SOME CHECKS FAILED: do not train yet, post the output above in the team chat."; exit 1; fi
