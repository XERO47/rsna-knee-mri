#!/bin/bash
# Architecture sweep over the frozen embeddings (fast: ~5 min each). Each run writes
# full_tx_<TAG>_summary.json + checkpoints + gold preds. Frozen features cap ~0.73, so this
# is about finding the best HEAD (pooling/size), not breaking the ceiling.
cd /workspace/code/fullcache || exit 1
BASE="FULL_EMB=/workspace/full_emb FULL_DATA=/workspace/data FULL_WORK=/workspace/out FULL_NW=8"
run(){ tag=$1; shift; echo "===== $tag :: $* ====="; env $BASE FULL_TAG=$tag "$@" python -u train_full_tx.py 2>&1 | tail -2; }

run cls_d384   FULL_POOL=cls  FULL_D=384 FULL_DEPTH=6  FULL_EPOCHS=40
run attn_d384  FULL_POOL=attn FULL_D=384 FULL_DEPTH=6  FULL_EPOCHS=40
run attn_d512  FULL_POOL=attn FULL_D=512 FULL_DEPTH=8  FULL_EPOCHS=40
run cls_d512   FULL_POOL=cls  FULL_D=512 FULL_DEPTH=8  FULL_EPOCHS=40
run mean_d512  FULL_POOL=mean FULL_D=512 FULL_DEPTH=6  FULL_EPOCHS=40
run attn_d768  FULL_POOL=attn FULL_D=768 FULL_DEPTH=8  FULL_EPOCHS=50 FULL_DROP=0.2
run cls_deep   FULL_POOL=cls  FULL_D=512 FULL_DEPTH=12 FULL_EPOCHS=50

echo "===== SWEEP SUMMARY ====="
cd /workspace/out && for f in full_tx_*_summary.json; do
  python3 - "$f" <<'PY' 2>/dev/null || true
import json,sys
d=json.load(open(sys.argv[1]))
c=d["config"]
print(f"{sys.argv[1]:40s} val {d['best_val_pool_auc']:.4f} | gold {d['gold_at_best_val_UNBIASED']:.4f} | oracle {d['best_gold_ORACLE_biased']:.4f} | {c['pool']} d{c['d']} L{c['depth']}")
PY
done
echo "SWEEP_DONE"
