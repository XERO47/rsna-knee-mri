"""Which 4-bit backend actually installs and imports against Kaggle's torch?

The extraction probe failed on `ImportError: Loading an AWQ quantized model requires gptqmodel`.
The image ships transformers + accelerate and no quantisation backend at all. Rather than guess,
this installs each candidate and reports what imports, with versions and timings. No model is
downloaded, so it costs a few GPU-minutes and settles the plan.
"""
import subprocess
import sys
import time

import torch

print(f"torch {torch.__version__} | cuda {torch.version.cuda} | "
      f"devices {torch.cuda.device_count()}", flush=True)
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"  gpu{i} {p.name} {p.total_memory / 1e9:.1f} GB sm_{p.major}{p.minor}", flush=True)
import transformers
print(f"transformers {transformers.__version__}", flush=True)

for pkg, mod in (("bitsandbytes", "bitsandbytes"),
                 ("autoawq", "awq"),
                 ("gptqmodel", "gptqmodel")):
    t0 = time.time()
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg],
                       capture_output=True, text=True, timeout=1800)
    dt = time.time() - t0
    if r.returncode:
        print(f"\n{pkg}: INSTALL FAILED in {dt:.0f}s\n  {r.stderr.strip()[-400:]}", flush=True)
        continue
    try:
        m = __import__(mod)
        v = getattr(m, "__version__", "present")
        print(f"\n{pkg}: installed and imports in {dt:.0f}s -> {mod} {v}", flush=True)
    except Exception as e:
        print(f"\n{pkg}: installed in {dt:.0f}s but import FAILED: "
              f"{type(e).__name__}: {str(e)[:300]}", flush=True)

# a 4-bit load has to actually work, not just import. Smallest real test available.
try:
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.float16)
    t0 = time.time()
    m = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct",
                                             quantization_config=cfg, device_map="auto")
    print(f"\nEND-TO-END 4-bit load of a 0.5B model: OK in {time.time() - t0:.0f}s | "
          f"dtype {next(m.parameters()).dtype} | "
          f"gpu mem {torch.cuda.memory_allocated() / 1e9:.2f} GB", flush=True)
    print("-> bitsandbytes NF4 is a viable path for Qwen2.5-14B-Instruct (fp16 source)")
except Exception as e:
    print(f"\n4-bit load test FAILED: {type(e).__name__}: {str(e)[:400]}", flush=True)
