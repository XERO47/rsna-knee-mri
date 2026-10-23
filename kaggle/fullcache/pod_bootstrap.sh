#!/bin/bash
# On-pod bootstrap for the full-cache TRAIN step: deps -> pull the 4 embed-group outputs
# from Kaggle (REST + aria2, ~1.5 GB total, fast) -> assemble into /workspace/full_emb.
# Training is launched separately in a held-open SSH (RUNBOOK: bg jobs die on disconnect).
set -u
CODE=/workspace/code/fullcache
EMBG=/workspace/emb_groups; OUT=/workspace/out
mkdir -p "$EMBG" "$OUT" /workspace/data
echo "[1/3] deps"
apt-get update -q >/dev/null && apt-get install -y -q aria2 >/dev/null
pip install -q pandas scikit-learn 2>/dev/null || true

echo "[2/3] pull 4 embed groups from Kaggle"
TOKEN=$(cat /root/.kaggle/access_token)
cd "$EMBG" || exit 1
for G in g0 g1 g2 g3; do
  SLUG="rsna-knee-embed-$G"
  curl -s -H "Authorization: Bearer $TOKEN" \
    "https://www.kaggle.com/api/v1/kernels/output?userName=zeroday19&kernelSlug=$SLUG&pageSize=60" -o /tmp/ko.json
  python3 -c "import json;d=json.load(open('/tmp/ko.json'));open('/tmp/f.tsv','w').write(''.join(x['fileName']+'\t'+x['url']+'\n' for x in d['files']))"
  echo "== $SLUG: $(wc -l </tmp/f.tsv) files =="
  while IFS=$'\t' read -r name url; do
    [ -z "$name" ] && continue
    aria2c -x16 -s16 -k1M --file-allocation=none --console-log-level=warn --max-tries=30 \
      --retry-wait=5 --allow-overwrite=true --header="Authorization: Bearer $TOKEN" -o "$name" "$url"
  done < /tmp/f.tsv
done

echo "[3/3] assemble -> /workspace/full_emb"
python3 "$CODE/assemble_emb.py" "$EMBG" /workspace/full_emb
echo "BOOTSTRAP_DONE $(date +%H:%M:%S)"
ls -la /workspace/full_emb
