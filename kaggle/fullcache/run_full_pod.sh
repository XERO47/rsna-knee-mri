#!/bin/bash
# End-to-end full-cache run on a fresh RunPod pod: stage -> embed -> train.
# Run inside tmux (detached launches silently failed on the image; see RUNBOOK).
# Prereqs you scp first (RUNBOOK 4a): /workspace/code/fullcache/*.py, /workspace/data
# (train_pool.csv, gold_test.csv), and /root/.kaggle/access_token (kernel scope).
set -eu
CODE=/workspace/code/fullcache
OUT=/workspace/out; mkdir -p "$OUT"
export FULL_ROOT=/workspace/full FULL_EMB_OUT=/workspace/full_emb
export FULL_EMB=/workspace/full_emb FULL_DATA=/workspace/data FULL_WORK="$OUT"

echo "[1/4] deps"
apt-get update -q && apt-get install -y -q aria2 tmux >/dev/null
pip install -q timm pandas scikit-learn pydicom 2>/dev/null || true

echo "[2/4] stage 16 shards (~160 GB) -> $FULL_ROOT"
if [ ! -f "$FULL_ROOT/slices_shard15.npy" ]; then
  bash "$CODE/stage_full.sh" > "$OUT/stage.log" 2>&1
fi
python -c "import sys;sys.path.insert(0,'$CODE');from load_full import FullCache;fc=FullCache('$FULL_ROOT');print('staged studies',len(fc.uids()),'shards',fc.shards)"

echo "[3/4] precompute embeddings (frozen encoder, every slice) -> $FULL_EMB_OUT"
if [ ! -f "$FULL_EMB_OUT/full_emb.npy" ]; then
  ( cd "$CODE" && python -u precompute_full_emb.py > "$OUT/precompute.log" 2>&1 )
  tail -2 "$OUT/precompute.log"
fi

echo "[4/4] train variable-length slice-transformer -> $OUT"
( cd "$CODE" && python -u train_full_tx.py > "$OUT/full_tx.log" 2>&1 )
echo "== summary =="; cat "$OUT/full_tx_summary.json" 2>/dev/null | python -c "import sys,json;d=json.load(sys.stdin);print('best val-pool',d['best_val_pool_auc'],'| final gold',d['final_gold_auc'])" || tail -5 "$OUT/full_tx.log"
touch "$OUT/FULL_PIPELINE_DONE"
echo "DONE $(date +%H:%M:%S)"
