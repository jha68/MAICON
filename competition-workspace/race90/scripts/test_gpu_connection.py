"""Run on a rented Pod: independent CUDA work on each of three assigned GPUs."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def gpu_worker():
    import torch
    torch.set_num_threads(1)
    torch.manual_seed(42)
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    device = torch.device("cuda:0")
    props = torch.cuda.get_device_properties(device)
    assert "A100" in props.name and props.total_memory > 79_000_000_000
    torch.backends.cuda.matmul.allow_tf32 = False
    a, b = torch.randn(128, 128), torch.randn(128, 128)
    reference = a @ b
    actual = (a.to(device) @ b.to(device)).cpu()
    torch.testing.assert_close(actual, reference, rtol=1e-4, atol=1e-4)
    x = torch.randn(64, 256, device=device)
    target = x @ torch.randn(256, 16, device=device)
    model = torch.nn.Sequential(torch.nn.Linear(256, 128), torch.nn.ReLU(), torch.nn.Linear(128, 16)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    losses = []
    for _ in range(30):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(model(x), target)
        assert torch.isfinite(loss)
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        optimizer.step()
        losses.append(float(loss.detach()))
    torch.cuda.synchronize()
    assert losses[-1] < losses[0]
    return {"gpu_slot": int(os.environ["CUDA_VISIBLE_DEVICES"]), "pid": os.getpid(),
            "name": props.name, "uuid": str(props.uuid), "memory_bytes": props.total_memory,
            "matmul_max_abs_error": float((actual-reference).abs().max()),
            "training_steps": 30, "initial_loss": losses[0], "final_loss": losses[-1],
            "torch": torch.__version__, "cuda": torch.version.cuda, "passed": True}


def main():
    if "--worker" in sys.argv:
        print(json.dumps(gpu_worker()))
        return
    import torch
    assert torch.cuda.device_count() == 3, torch.cuda.device_count()
    started = time.time()
    def one(slot):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(slot), OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
        result = subprocess.run([sys.executable, __file__, "--worker"], env=env, check=True,
                                capture_output=True, text=True, timeout=120)
        return json.loads(result.stdout.strip())
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        workers = list(pool.map(one, range(3)))
    assert len({w["uuid"] for w in workers}) == 3
    report = {"passed": all(w["passed"] for w in workers), "gpu_count": 3,
              "parallel_workers": workers, "elapsed_seconds": time.time() - started,
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    Path("gpu-test-result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
