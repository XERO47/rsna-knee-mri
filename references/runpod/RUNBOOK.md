# RunPod Runbook — training when Kaggle GPU quota is exhausted

Battle-tested end-to-end procedure for renting a RunPod GPU, staging the Kaggle
image caches onto it, training, and retrieving results. Every step here encodes a
failure we actually hit on 2026-10-08 and its fix. Follow the **Golden Path** and
you avoid all of them.

> Why this exists: Kaggle gives 30 GPU-hours/week. When that's exhausted, training
> moves to RunPod. Staging the 12–32 GB caches from Kaggle to the pod is the hard
> part — it broke in ~6 different ways before it worked. This is the recipe that works.

---

## 0. TL;DR — the Golden Path (copy this order)

1. **Create a SECURE-cloud pod** (never community) with the **dockerArgs SSH-key
   injection** (the image ignores `PUBLIC_KEY`).
2. **SSH in** on the mapped public `ip:port`, with keepalives.
3. **Install deps**: `apt-get install -y aria2 tmux` + the pip packages.
4. **Copy BOTH Kaggle creds** to the pod: `~/.kaggle/kaggle.json` **and**
   `~/.kaggle/access_token` (the token has kernel scope; the json does not).
5. **Stage caches via REST + aria2**, NOT the kaggle CLI (its `kernels output` is
   broken): hit the REST endpoint with the OAuth **Bearer token** to get signed
   URLs, download each with **`aria2c -x16`** (16 connections beats the throttle).
6. **Run training in a held-open foreground SSH** (`run_in_background` on the
   caller side). This image **kills background jobs on disconnect** — `setsid`,
   `nohup`, and `tmux` all die. Tee to a log; poll the log/sentinel.
7. **scp the outputs back**, then **stop or terminate** the pod to end billing.

Secrets: the RunPod API key and Kaggle token are secrets — keep them on the pod /
in a session scratch file, never in the repo or a commit. Rotate the API key after.

---

## 1. Create the pod (SECURE cloud, SSH injected)

**Use SECURE cloud, not COMMUNITY.** Community hosts gave: constant SSH drops
(`exit 255`), download stalls, and — fatally — **you lose the GPU if you `podStop`**
("not enough free GPUs" on resume). Secure (~$0.74/hr for a 4090) had rock-solid
SSH and 450 MB/s downloads. The extra ~$0.40/hr is worth it.

RunPod GraphQL endpoint (use **curl**, not python-urllib — Cloudflare 403s the
urllib user-agent). Endpoint: `https://api.runpod.io/graphql?api_key=$RUNPOD_API_KEY`.

The runpod/pytorch image does **not** honor the `PUBLIC_KEY` env for SSH, and the
account-key + `ssh.runpod.io` proxy also failed. What works: a `dockerArgs` startup
that injects your key and starts sshd:

```bash
# dockerArgs value (bash -c '...'); $PUBLIC_KEY is expanded on the pod at runtime
bash -c 'mkdir -p /root/.ssh && echo "$PUBLIC_KEY" >> /root/.ssh/authorized_keys \
  && chmod 700 /root/.ssh && chmod 600 /root/.ssh/authorized_keys \
  && ssh-keygen -A >/dev/null 2>&1; (service ssh start || /usr/sbin/sshd); sleep infinity'
```

Create mutation (build the JSON with python to escape the pubkey + dockerArgs,
then POST with curl `--data @file.json`):

```graphql
mutation($pub:String!,$d:String!){
  podFindAndDeployOnDemand(input:{
    cloudType: SECURE, gpuCount:1, volumeInGb:0, containerDiskInGb:140,
    minVcpuCount:8, minMemoryInGb:32,
    gpuTypeId:"NVIDIA GeForce RTX 4090",     # fallback: "NVIDIA GeForce RTX 3090"
    name:"knee-train", imageName:"runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04",
    dockerArgs:$d, ports:"22/tcp", env:[{key:"PUBLIC_KEY", value:$pub}]
  }){ id desiredStatus costPerHr }
}
```

- `containerDiskInGb:140` — the vol3d cache is ~11 GB, cache64 ~32 GB; leave headroom.
- On `SUPPLY_CONSTRAINT` ("no instances available"), retry or fall back GPU types:
  4090 → 3090 Ti → 3090 → A5000.
- A fresh SSH keypair: `ssh-keygen -t ed25519 -f knee_rp -N ""`.

---

## 2. Connect (SSH)

Poll for the mapped public port (privatePort 22 → public `ip:publicPort`):

```graphql
query{ pod(input:{podId:"<ID>"}){ runtime{ ports{ ip isIpPublic privatePort publicPort } } } }
```

Then SSH with keepalives (reduces the drops):

```bash
SSH="-i knee_rp -p <PORT> -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
     -o ServerAliveInterval=15 -o TCPKeepAlive=yes -o ConnectTimeout=25"
ssh $SSH root@<IP> 'nvidia-smi -L; nproc'
```

---

## 3. Install dependencies

```bash
apt-get update && apt-get install -y aria2 tmux      # aria2 is MANDATORY for staging
pip install -q timm kaggle iopath fvcore pandas scikit-learn
pip install -q --no-deps pytorchvideo                # for X3D; pulls no torch downgrade
# sanity: python -c "import timm,pandas; from pytorchvideo.models.hub import x3d_m; x3d_m(pretrained=True)"
```

The base image is missing `pandas`/`scikit-learn` — the trainers need them. `aria2`
needs `apt-get update` first or it reports "Unable to locate package".

---

## 4. Stage Kaggle data — the part that breaks

### 4a. Credentials: copy BOTH files

```bash
scp -i knee_rp -P <PORT> ~/.kaggle/kaggle.json ~/.kaggle/access_token root@<IP>:/root/.kaggle/
ssh $SSH root@<IP> 'chmod 600 /root/.kaggle/*'
```

- `kaggle.json` (the API key) has **DATASET scope only** → `kernels/output` returns
  **403**. Regenerating the token does NOT fix this.
- `~/.kaggle/access_token` is the local **OAuth** token and has **kernel scope** —
  this is what lets you pull kernel outputs. Your *local* kaggle works because it
  uses this token, not the json.

### 4b. The kaggle CLI is broken — use REST + aria2

`kaggle kernels output` in CLI **2.2.4 creates 0-byte files** (hangs). Do NOT use it.
Instead, hit the REST endpoint with the **Bearer OAuth token** to get signed
`kaggleusercontent.com` URLs, then download with **aria2 (16 connections)** — a
single connection is throttled to ~91 KB/s; 16 connections hit **~450 MB/s**.

```bash
# on the pod
TOKEN=$(cat /root/.kaggle/access_token)
curl -s -H "Authorization: Bearer $TOKEN" \
  "https://www.kaggle.com/api/v1/kernels/output?userName=<USER>&kernelSlug=<SLUG>&pageSize=60" \
  -o /tmp/ko.json
# files[] each have .fileName and a signed .url -> download each:
aria2c -x16 -s16 -k1M --file-allocation=none --max-tries=30 --retry-wait=5 \
  --header="Authorization: Bearer $TOKEN" -d /workspace/data/<dir> -o "<fileName>" "<url>"
```

A ready script that does the whole thing (fetch list → aria2 each file) is in
`scripts/` of your session scratch as `dl_loop.sh`; the key points it encodes:

- **Trailing-newline bug**: a `while IFS= read` loop over a `'\n'.join()` list
  **skips the last file** (no trailing newline). Add a trailing newline, or download
  the last file explicitly, or you'll be missing e.g. `vol_sagittal_shard0.npy`.
- **Re-fetch the URL list if a download errors** — signed URLs expire (~1 h).
- **`--lowest-speed-limit=500K`** + a resume loop (`--continue`) survives a stall.

### 4c. Datasets (labels etc.) are easier

Dataset downloads work with the token (dataset scope). But the label CSVs
(`gold_test.csv`, `train_pool.csv`) are tiny and local — just `scp` them:

```bash
scp -i knee_rp -P <PORT> data/processed/train_pool.csv data/processed/gold_test.csv \
    root@<IP>:/workspace/data/labels/
```

### 4d. Verify the cache

```bash
# plane volumes are uint8, (N, D, H, W); ids are object dtype (load with allow_pickle, NOT mmap)
python3 -c "import numpy as np; a=np.load('vol_axial_shard0.npy',mmap_mode='r'); print(a.shape,a.dtype)"
```

**`du` lies on sparse files** (aria2 writes to offsets). Don't trust `du -sh` for
progress — read aria2's own `DL:` speed line, or compare file sizes to expected.

---

## 5. Run training — it must stay in the foreground

**This image kills session processes on SSH disconnect.** `setsid`, `nohup`, AND
`tmux new-session -d` all die when the SSH that launched them closes (tmux server
socket disappears). Symptoms: you "launch" and `procs: 0`, and you keep re-reading a
**stale log** from the first run, thinking it's still crashing.

**The fix:** run training in the **foreground of a held-open SSH** — i.e. a
`run_in_background` SSH on the caller side that stays connected for the whole run:

```bash
# caller side: run_in_background = true
ssh $SSH root@<IP> 'cd /workspace/code && \
  VOL_DATA=/workspace/data VOL_WORK=/workspace/out VOL_EPOCHS=18 VOL_BATCH=16 VOL_WORKERS=16 \
  python -u train_vol3d_v2.py 2>&1 | tee /workspace/out/train.log; echo EXIT=$?'
```

- `python -u` (unbuffered) so the stream is live.
- `tee` to a log so progress survives even if your view drops.
- Keepalives (`ServerAliveInterval`) keep the connection up for a ~30–60 min run.
- **Smoke-test first** with a tiny cap (e.g. `VOL_MAX_STUDIES=40 VOL_EPOCHS=1`) run
  *directly* (not backgrounded) so you see the real output and catch bugs in <60 s.
  (That's how we found the `roc_auc_score` continuous-label crash.)
- Monitor by polling the log (`grep "epoch\|COMPLETE\|Traceback"`), not process state.

---

## 6. Retrieve outputs

```bash
scp -i knee_rp -P <PORT> root@<IP>:/workspace/out/vol3d_v2_gold_pred.npy \
    root@<IP>:/workspace/out/vol3d_v2_summary.json ./out/
```

Download everything you need **before** stopping the pod.

---

## 7. Stop / terminate (end billing)

- `podStop` keeps the container disk (deps + staged cache survive) but keeps a small
  storage charge. **On community it often can't resume** (GPU taken). On secure it's
  safer but not guaranteed.
- `podTerminate` deletes everything. Since staging is now fast (~1 min at 450 MB/s)
  and deps install in a few min, **terminate + recreate is a perfectly fine pattern**
  if resume is unreliable.
- Mutations: `podStop(input:{podId:"<ID>"}){id desiredStatus}`,
  `podTerminate(input:{podId:"<ID>"})`, `podResume(input:{podId:"<ID>",gpuCount:1})`.

Always `podStop`/`podTerminate` when done — a forgotten pod bills 24/7.

---

## Gotchas quick-reference

| Symptom | Cause | Fix |
|---|---|---|
| `podFindAndDeployOnDemand` 403 code 1010 | python-urllib UA blocked by Cloudflare | use `curl` |
| SSH `Permission denied (publickey)` | image ignores `PUBLIC_KEY` env | inject key via `dockerArgs` |
| SSH `exit 255` mid-command, output truncated | community host flakiness | use **SECURE** cloud |
| Can't `podResume` ("no free GPUs") | stopped a **community** pod, GPU taken | don't stop community; terminate+recreate |
| `kernels/output` → 403 | kaggle.json has dataset scope only | use `access_token` (Bearer, OAuth) |
| `kaggle kernels output` → 0-byte files | CLI 2.2.4 bug | use REST + aria2 |
| download ~91 KB/s / stalls | per-connection CDN throttle | `aria2c -x16 -s16` |
| last cache file missing | `while read` drops no-trailing-newline last line | add trailing newline / fetch it explicitly |
| `du -sh` shows wrong size | aria2 sparse writes | read aria2 `DL:` rate, not du |
| "launched" but `procs: 0`, stale log | image kills bg jobs on disconnect | foreground, held-open SSH |
| `roc_auc_score: continuous format` | passing soft labels as y_true | binarize `(y>=0.5)` before AUC |
| `No module named pandas/sklearn` | base image missing them | `pip install pandas scikit-learn` |
| `Unable to locate package aria2` | stale apt index | `apt-get update` first |

---

## Appendix — minimal data-staging script (REST + aria2)

```bash
#!/bin/bash
# stage one kernel's output files to a dir, fast + resumable.
# usage: bash stage.sh <USER> <SLUG> <DEST_DIR>
U="$1"; SLUG="$2"; DEST="$3"; mkdir -p "$DEST"; cd "$DEST" || exit 1
TOKEN=$(cat /root/.kaggle/access_token)
curl -s -H "Authorization: Bearer $TOKEN" \
  "https://www.kaggle.com/api/v1/kernels/output?userName=$U&kernelSlug=$SLUG&pageSize=200" -o /tmp/ko.json
python3 - <<'PY'
import json
d=json.load(open("/tmp/ko.json"))
open("/tmp/files.tsv","w").write("".join(f"{x['fileName']}\t{x['url']}\n" for x in d["files"]))  # trailing \n per line!
PY
while IFS=$'\t' read -r name url; do
  [ -z "$name" ] && continue
  echo "== $name =="
  aria2c -x16 -s16 -k1M --file-allocation=none --continue=true --max-tries=30 \
    --retry-wait=5 --lowest-speed-limit=500K --allow-overwrite=false \
    --header="Authorization: Bearer $TOKEN" -o "$name" "$url"
done < /tmp/files.tsv
echo "DONE"; ls -la
```

---

*Companion: the agent-facing condensed version lives in Claude memory as
`runpod-staging-playbook`. Keep both in sync if the procedure changes.*
