"""A minimal GPU kernel with NO attached inputs.

Diagnostic only: arm31probe and arm31phase have sat QUEUED for hours with 46 GB and 64 GB of
kernel_sources respectively. If this runs promptly, the queue is about input provisioning and
those kernels need restructuring; if this also queues, it is Kaggle GPU capacity and waiting is
the only option.
"""
import time

import torch

print(f"cuda available {torch.cuda.is_available()} | devices {torch.cuda.device_count()}",
      flush=True)
for i in range(torch.cuda.device_count()):
    print(f"  {i}: {torch.cuda.get_device_name(i)}", flush=True)
t0 = time.time()
x = torch.randn(4096, 4096, device="cuda")
for _ in range(20):
    x = x @ x.T / 4096.0
torch.cuda.synchronize()
print(f"20 matmuls of 4096^2 in {time.time() - t0:.2f}s -- the GPU is real and usable",
      flush=True)
