import socket
import time
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import train_armc_roi_kaggle as trainer


class SyntheticStudies(Dataset):
    """Small deterministic bags that exercise the trainer's real I/O shape."""

    def __init__(self, rank: int):
        self.rank = rank

    def __len__(self):
        return 2

    def __getitem__(self, index):
        generator = torch.Generator().manual_seed(1000 + self.rank * 10 + index)
        global_x = torch.randint(
            0,
            256,
            (2, 3, 16, 16),
            dtype=torch.uint8,
            generator=generator,
        )
        global_mask = torch.ones(2, dtype=torch.bool)
        roi_x = {
            target: torch.randint(
                0,
                256,
                (1, 3, 16, 16),
                dtype=torch.uint8,
                generator=generator,
            )
            for target in trainer.ROI_TARGETS
        }
        roi_mask = {
            target: torch.ones(1, dtype=torch.bool)
            for target in trainer.ROI_TARGETS
        }
        label = torch.full(
            (len(trainer.LABELS),),
            float((self.rank + index) % 2),
        )
        return global_x, global_mask, roi_x, roi_mask, label


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def smoke_worker(rank: int, world: int, port: int) -> None:
    trainer._RANK, trainer._WORLD = rank, world
    trainer.ARCH = "tiny_test"
    trainer.RES = 16
    trainer.WORKERS = 0
    trainer.PROGRESS = False
    trainer.CHANNELS_LAST = False
    torch.manual_seed(7)

    dist.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world,
    )
    try:
        probe = torch.ones(1)
        dist.all_reduce(probe)
        if int(probe.item()) != world:
            raise AssertionError(f"collective returned {probe.item()}, expected {world}")

        loader_options = trainer.make_loader_kwargs(torch.device("cpu"))
        if loader_options["num_workers"] != 0:
            raise AssertionError("DDP smoke loader unexpectedly creates child processes")
        if "persistent_workers" in loader_options or "worker_init_fn" in loader_options:
            raise AssertionError("zero-worker loader received worker-only options")
        loader = DataLoader(
            SyntheticStudies(rank),
            batch_size=1,
            shuffle=False,
            **loader_options,
        )

        model = trainer.ArmC()
        baseline = trainer.evaluate(model, loader, torch.device("cpu"), "smoke baseline")
        if baseline["combined_auc"] != baseline["global_auc"]:
            raise AssertionError("zero-initialized ROI heads changed the baseline")

        trainer.set_training_phase(model, trainer.FREEZE_GLOBAL_EPOCHS)
        wrapped = trainer.wrap_for_training(model, rank, world, torch.device("cpu"))
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        gx, gm, rx, rm, labels = next(iter(loader))
        logits, _ = wrapped(gx, gm, rx, rm)
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        weights = torch.cat(
            [parameter.detach().reshape(-1).float() for parameter in model.parameters()]
        )
        gathered = [torch.empty_like(weights) for _ in range(world)]
        dist.all_gather(gathered, weights)
        for peer in gathered[1:]:
            torch.testing.assert_close(gathered[0], peer, rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


class DDPTrainingSmokeTest(unittest.TestCase):
    def test_two_rank_loader_evaluation_and_training_step(self):
        world = 2
        context = mp.start_processes(
            smoke_worker,
            args=(world, free_port()),
            nprocs=world,
            join=False,
            start_method="spawn",
        )
        deadline = time.monotonic() + 60
        while not context.join(timeout=1):
            if time.monotonic() >= deadline:
                for process in context.processes:
                    if process.is_alive():
                        process.terminate()
                self.fail("two-rank DDP smoke test exceeded 60 seconds")


if __name__ == "__main__":
    unittest.main()
