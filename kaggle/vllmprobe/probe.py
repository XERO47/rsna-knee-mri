"""Can Kaggle's T4 run the CUDA-13 vLLM stack? (py3.13, driver TBD)

v2: `pip install vllm` -> vllm 0.30.0 + torch 2.13.0+cu130, failed ONLY on torchaudio still being
    cu128 (a version-check mismatch, not a missing runtime).
v3: pinning torch 2.11+cu128 -> vllm 0.26.0, but its kernels need libcudart.so.13 (CUDA 13). So every
    modern vLLM wheel is CUDA-13; Kaggle's image is CUDA 12.8. The only way forward on Kaggle is to go
    fully CUDA 13 -- IF the T4 driver supports it.
v4: install vllm (lets torch go to 2.13+cu130), ALIGN torchaudio+torchvision to cu130, report the GPU
    driver, then test TP=1/TP=2. All findings -> /kaggle/working/vllm_probe_result.json.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

RESULT = Path("/kaggle/working/vllm_probe_result.json")
CU130 = "https://download.pytorch.org/whl/cu130"
R = {"stage": "start"}


def save():
    try:
        RESULT.write_text(json.dumps(R, indent=1, default=str), encoding="utf-8")
    except Exception:
        pass


def run(cmd, timeout=2400):
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return r.returncode, r.stdout, r.stderr


rc, out, err = run(["nvidia-smi", "--query-gpu=driver_version,name,memory.total",
                    "--format=csv,noheader"], timeout=120)
R["driver"] = (out + err).strip()
# CUDA 13 user-mode needs driver >= 580; capture the raw nvidia-smi header too
rc, out, err = run(["nvidia-smi"], timeout=120)
R["nvidia_smi_head"] = "\n".join((out + err).splitlines()[:4])
R["stage"] = "install_vllm"
save()

t0 = time.time()
rc, out, err = run([sys.executable, "-m", "pip", "install", "-q", "vllm"])
R["install_vllm"] = {"rc": rc, "seconds": round(time.time() - t0), "err_tail": err[-1500:] if rc else ""}
save()

# Align the rest of the torch family to whatever CUDA torch now uses (cu130).
t0 = time.time()
rc, out, err = run([sys.executable, "-m", "pip", "install", "-q", "--upgrade",
                    "torchaudio", "torchvision", "--extra-index-url", CU130])
R["align_family"] = {"rc": rc, "seconds": round(time.time() - t0), "err_tail": err[-1500:] if rc else ""}
save()

rc, out, err = run([sys.executable, "-c",
                    "import torch;print(torch.__version__, torch.version.cuda)"])
R["torch_after"] = (out + err).strip()
rc, out, err = run([sys.executable, "-m", "pip", "show", "vllm"])
R["vllm_show"] = out.strip()[:200]
R["stage"] = "generate"
save()

test = (
    "import sys, time\n"
    "tp = int(sys.argv[1])\n"
    "from vllm import LLM, SamplingParams\n"
    "t0 = time.time()\n"
    "llm = LLM(model='Qwen/Qwen2.5-0.5B-Instruct', tensor_parallel_size=tp, dtype='float16',\n"
    "          max_model_len=2048, gpu_memory_utilization=0.85, enforce_eager=True,\n"
    "          disable_custom_all_reduce=True)\n"
    "o = llm.generate(['Say OK.'], SamplingParams(max_tokens=5, temperature=0))\n"
    "print('GENTEXT', repr(o[0].outputs[0].text), 'SECS', round(time.time()-t0))\n")
Path("/kaggle/working/vllm_test.py").write_text(test, encoding="utf-8")

R["generate"] = {}
for tp in (1, 2):
    t0 = time.time()
    rc, out, err = run([sys.executable, "/kaggle/working/vllm_test.py", str(tp)], timeout=1800)
    R["generate"][f"tp{tp}"] = {"rc": rc, "ok": rc == 0 and "GENTEXT" in out,
                                "seconds": round(time.time() - t0), "tail": (out + err)[-2500:]}
    save()

R["stage"] = "complete"
save()
print("VLLM_PROBE_V4_COMPLETE", flush=True)
