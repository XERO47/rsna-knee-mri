#!/bin/bash
# Stage the full-cache (16 shards, ~144 GB) from Kaggle kernel outputs -> /workspace/full,
# via REST + aria2 (the kaggle CLI's `kernels output` is broken; access_token has kernel
# scope, kaggle.json does not). All shards land in ONE dir so load_full.FullCache reads them.
set -u
DEST=${FULL_DEST:-/workspace/full}
mkdir -p "$DEST" && cd "$DEST" || exit 1
TOKEN=$(cat /root/.kaggle/access_token)
for S in $(seq 0 15); do
  SLUG="rsna-knee-fullcache-s$S"
  curl -s -H "Authorization: Bearer $TOKEN" \
    "https://www.kaggle.com/api/v1/kernels/output?userName=zeroday19&kernelSlug=$SLUG&pageSize=60" -o /tmp/ko.json
  python3 -c "import json;d=json.load(open('/tmp/ko.json'));open('/tmp/f.tsv','w').write(''.join(x['fileName']+'\t'+x['url']+'\n' for x in d['files']))"
  echo "== $SLUG: $(wc -l </tmp/f.tsv) files =="
  while IFS=$'\t' read -r name url; do
    [ -z "$name" ] && continue
    echo "GET $name"
    aria2c -x16 -s16 -k1M --file-allocation=none --console-log-level=warn --max-tries=30 --retry-wait=5 \
      --continue=true --auto-file-renaming=false --allow-overwrite=false \
      --header="Authorization: Bearer $TOKEN" -o "$name" "$url"
  done < /tmp/f.tsv
done
echo "DONE $(date +%H:%M:%S)"
ls -la "$DEST" | grep -c slices_shard
python3 -c "import glob;print('shards present:',sorted(int(p.split('shard')[1].split('.')[0]) for p in glob.glob('$DEST/slices_shard*.npy')))"
