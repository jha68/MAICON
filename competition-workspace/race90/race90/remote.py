"""Runpod lease + SSH transport. This module never provisions on import."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import importlib.metadata
import ipaddress
import json
import os
import shlex
import shutil
import subprocess
import sys
import tarfile
import threading
import time
import urllib.request
import uuid
from .bundle import source_zip, safe_extract_tar
from .io import atomic_json
from .process import run_command


def api_key():
    value = os.environ.get("RUNPOD_API_KEY")
    if value:
        return value
    dotenv = Path.cwd() / ".env"
    if dotenv.exists():
        for line in dotenv.read_text().splitlines():
            line = line.strip()
            if line.startswith("export "):
                line = line[7:].lstrip()
            if line.startswith("RUNPOD_API_KEY="):
                value = line.split("=", 1)[1].strip().strip("\"'")
                if value:
                    return value
    path = Path.home() / ".runpod/config.toml"
    if path.exists():
        try:
            try:
                import tomllib
            except ImportError:
                import tomli as tomllib
            data = tomllib.loads(path.read_text())
            for obj in (data, data.get("default", {}), data.get("runpod", {})):
                for key in ("api_key", "apiKey", "apikey"):
                    if obj.get(key):
                        return obj[key]
        except ImportError:
            pass
    raise ValueError("Set RUNPOD_API_KEY in the environment before enabling Runpod")


def graphql(query, variables):
    payload = json.dumps({"query": query, "variables": variables}).encode()
    request = urllib.request.Request("https://api.runpod.io/graphql", data=payload,
                                     headers={"Authorization": "Bearer " + api_key(), "Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=30) as response:
        result = json.load(response)
    if result.get("errors"):
        raise RuntimeError("; ".join(e["message"] for e in result["errors"]))
    return result["data"]


def eligible_gpus(catalog, priorities, cloud, gpu_count, max_gpu_hourly):
    if isinstance(catalog, dict):
        catalog = catalog.get("gpus", catalog.get("data", []))
    price_key = "securePricePerHr" if cloud == "SECURE" else "communityPricePerHr"
    eligible = []
    for ident in priorities:
        entry = next((g for g in catalog if ident in (g.get("id"), g.get("gpuId"), g.get("displayName"))), None)
        if entry is None:
            continue
        price = entry.get(price_key)
        if price is not None and 0 < float(price) * gpu_count <= max_gpu_hourly:
            eligible.append((entry.get("id", entry.get("gpuId")), float(price) * gpu_count))
    return eligible


class Runpod:
    def __init__(self, cfg, run_dir, compute_deadline, cancel):
        self.cfg, self.options = cfg, cfg["runpod"]
        self.run_dir = Path(run_dir)
        self.deadline = compute_deadline
        self.cancel = cancel
        self.pod_id = None
        self.owned = False
        self.host = None
        self.port = None
        self.closed = False
        self.lock = threading.Lock()
        self.dependency_lock = threading.Lock()
        self.uploaded_dependencies = {}
        self.cli = self.options.get("cli", "runpodctl")
        self.key_file = str(Path(self.options.get("ssh_key_path", "~/.ssh/id_ed25519")).expanduser())
        self.remote_root = "/workspace/race90-" + uuid.uuid4().hex[:12]
        self.state_file = self.run_dir / "runpod.json"
        if self.state_file.exists():
            old = json.loads(self.state_file.read_text())
            if old.get("pod_id") and old.get("owned") and old.get("status") not in {"deleted"}:
                self.options = {**self.options, "existing_pod_id": old["pod_id"]}
                self.pod_id, self.owned = old["pod_id"], True

    def ctl(self, *args, deadline=None):
        out = run_command([self.cli, *args], deadline or self.deadline, capture=True, cancel=self.cancel)
        return json.loads(out)

    @property
    def remote_python(self):
        return self.remote_root + "/.venv/bin/python"

    def record(self, **extra):
        previous = json.loads(self.state_file.read_text()) if self.state_file.exists() else {}
        state = {**previous, "pod_id": self.pod_id, "owned": self.owned, "remote_root": self.remote_root,
                 "updated_at": time.time(), **extra}
        atomic_json(self.state_file, state)

    def ssh_args(self):
        return ["ssh", "-i", self.key_file, "-p", str(self.port), "-o", "BatchMode=yes",
                "-o", "IdentitiesOnly=yes", "-o", "StrictHostKeyChecking=accept-new",
                "-o", "ConnectTimeout=10", "-o", "ServerAliveInterval=15",
                "-o", "ServerAliveCountMax=2", "root@" + self.host]

    def ssh(self, argv, deadline=None, log=None, capture=False, cancellable=True):
        # OpenSSH sends one shell command; quote every argument, including paths.
        return run_command(self.ssh_args() + [shlex.join(argv)], deadline or self.deadline,
                           log=log, capture=capture, cancel=self.cancel if cancellable else None)

    def copy(self, src, dst, deadline=None, cancellable=True):
        return run_command(["scp", "-i", self.key_file, "-P", str(self.port), "-o", "BatchMode=yes",
                            "-o", "IdentitiesOnly=yes", "-o", "StrictHostKeyChecking=accept-new",
                            "-o", "ConnectTimeout=10", str(src), str(dst)],
                           deadline or self.deadline, log=self.run_dir / "transfer.log",
                           cancel=self.cancel if cancellable else None)

    def prepare(self):
        setup_end = min(self.deadline, time.time() + self.options.get("setup_timeout_seconds", 420))
        try:
            api_key()  # Check auth before any infra action.
            self.ctl("user", deadline=setup_end)
            version = run_command([self.cli, "version"], setup_end, capture=True, cancel=self.cancel).strip()
            if not Path(self.key_file).is_file():
                raise ValueError(f"SSH key missing: {self.key_file}; run runpodctl doctor before the competition")
            if self.options.get("existing_pod_id"):
                self.pod_id = self.options["existing_pod_id"]
            else:
                self.create(setup_end)
            self.record(status="starting", cli_version=version)
            self.wait_ssh(setup_end)
            self.ssh(["mkdir", "-p", self.remote_root], deadline=setup_end)
            self.upload_and_install(setup_end)
            probe = self.ssh([self.remote_python, "-c", "import torch,json; print(json.dumps({'cuda':torch.cuda.is_available(),'gpus':torch.cuda.device_count()}))"],
                             deadline=setup_end, capture=True)
            health = json.loads(probe.strip().splitlines()[-1])
            if not health["cuda"] or health["gpus"] < int(self.options.get("gpu_count", 1)):
                raise RuntimeError(f"CUDA probe failed: {health}")
            self.record(status="ready", health=health, cli_version=version)
            return self
        except BaseException as e:
            self.record(status="failed", error=str(e))
            self.cleanup()
            raise

    def create(self, setup_end):
        options = self.options
        cap = float(options.get("max_gpu_hourly_usd", 0))
        if cap <= 0:
            raise ValueError("Set runpod.max_gpu_hourly_usd explicitly; cap covers GPU compute, excludes storage")
        count = int(options.get("gpu_count", 1))
        if count < 1:
            raise ValueError("gpu_count must be positive")
        cloud = options.get("cloud_type", "SECURE")
        if cloud not in {"SECURE", "COMMUNITY"}:
            raise ValueError("cloud_type must be SECURE or COMMUNITY")
        catalog = self.ctl("gpu", "list", "--include-unavailable", deadline=setup_end)
        priorities = options.get("gpu_priorities", ["NVIDIA A100-SXM4-80GB", "NVIDIA A100 80GB PCIe"])
        eligible = eligible_gpus(catalog, priorities, cloud, count, cap)
        if not eligible:
            raise RuntimeError("No configured GPU fits the live catalog and price cap")
        name = "race90-" + self.run_dir.name
        lease_end = time.time() + float(options.get("lease_minutes", self.cfg["total_minutes"] + 10)) * 60
        for gpu, hourly in eligible:
            if time.time() >= setup_end or self.cancel.is_set():
                raise TimeoutError("GPU setup budget expired")
            payload = {"name": name, "cloudType": cloud, "gpuTypeId": gpu, "gpuCount": count,
                       "startSsh": True, "supportPublicIp": True, "ports": "22/tcp",
                       "containerDiskInGb": int(options.get("container_disk_gb", 40)),
                       "volumeInGb": int(options.get("volume_gb", 0)), "volumeMountPath": "/workspace",
                       "terminateAfter": datetime.fromtimestamp(lease_end, timezone.utc).isoformat()}
            if options.get("template_id"):
                payload["templateId"] = options["template_id"]
            else:
                payload["imageName"] = options.get("image", "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404")
            for src, dst in (("data_center_id", "dataCenterId"), ("network_volume_id", "networkVolumeId")):
                if options.get(src):
                    payload[dst] = options[src]
            # Live runpodctl 2.14 has no --terminate-after flag. Use the documented
            # GraphQL creation field; all catalog/SSH/lifecycle work stays in CLI.
            query = "mutation($input: PodFindAndDeployOnDemandInput!) { podFindAndDeployOnDemand(input:$input) { id name } }"
            try:
                result = graphql(query, {"input": payload})["podFindAndDeployOnDemand"]
                self.pod_id = result["id"]
            except RuntimeError as e:
                if any(token in str(e).lower() for token in ("no available", "not enough", "insufficient capacity", "no instances")):
                    continue
                raise
            except (OSError, TimeoutError):
                # A timed-out mutation may have succeeded. Reconcile by our unique
                # name, never issue another create after an ambiguous response.
                pods = self.ctl("pod", "list", "--all", deadline=setup_end)
                if isinstance(pods, dict):
                    pods = pods.get("pods", pods.get("data", []))
                matches = [p for p in pods if p.get("name") == name]
                if len(matches) != 1:
                    self.record(status="creation_uncertain", pod_name=name,
                                error="Check this pod name in the console; automatic create retry disabled")
                    raise RuntimeError(f"Creation response ambiguous; check Runpod console for {name}")
                self.pod_id = matches[0]["id"]
            self.owned = True
            self.record(status="created", gpu=gpu, gpu_hourly_usd=hourly, lease_end=lease_end)
            env = os.environ.copy()
            env["RUNPOD_API_KEY"] = api_key()
            env["PYTHONPATH"] = os.pathsep.join([str(Path(__file__).parent.parent), self.cfg["project_root"], env.get("PYTHONPATH", "")])
            with (self.run_dir / "lease.log").open("ab") as log:
                subprocess.Popen([sys.executable, "-m", "race90", "_lease", self.cli, self.pod_id,
                                  str(lease_end), str(self.run_dir / "lease-cancelled")],
                                 start_new_session=True, stdout=log, stderr=subprocess.STDOUT, env=env)
            return
        raise RuntimeError("No available GPU among configured priorities")

    def wait_ssh(self, setup_end):
        while time.time() < setup_end and not self.cancel.is_set():
            pod = self.ctl("pod", "get", self.pod_id, deadline=setup_end)
            info = pod.get("ssh", {})
            if info.get("ip") and info.get("port"):
                ipaddress.ip_address(info["ip"])
                self.host, self.port = info["ip"], int(info["port"])
                try:
                    self.ssh(["true"], deadline=min(setup_end, time.time() + 15), capture=True)
                    return
                except (RuntimeError, TimeoutError):
                    pass
            self.cancel.wait(2)
        raise TimeoutError("Pod SSH did not become usable within setup budget")

    def upload_and_install(self, setup_end):
        source = self.run_dir / "source.zip"
        source.write_bytes(source_zip(self.cfg))
        data = self.run_dir / "data.tar"
        root = Path(self.cfg["data_root"])
        # No compression for JPEG/PNG/zip data; avoid wasting CPU on recoding.
        with tarfile.open(data, "w") as archive:
            for path in root.rglob("*"):
                if self.cancel.is_set() or time.time() >= setup_end:
                    raise TimeoutError("Dataset packing exceeded setup budget")
                if path.is_symlink():
                    raise ValueError(f"Dataset symlinks must be replaced with real files: {path}")
                if path.is_file():
                    archive.add(path, arcname=str(Path("data") / path.relative_to(root)), recursive=False)
        self.copy(source, f"root@{self.host}:{self.remote_root}/source.zip", deadline=setup_end)
        self.ssh(["python", "-m", "zipfile", "-e", self.remote_root + "/source.zip", self.remote_root], deadline=setup_end)
        # Official images can mark system Python externally managed (PEP 668).
        # Reuse their CUDA/PyTorch packages without modifying system Python.
        self.ssh(["python", "-m", "venv", "--system-site-packages", self.remote_root + "/.venv"], deadline=setup_end)
        packages = ["numpy", "pandas", "pillow", "PyYAML", "joblib",
                    "scikit-learn==" + importlib.metadata.version("scikit-learn")]
        if any(t["kind"] == "yolo" for t in self.cfg["tasks"]):
            packages.append("ultralytics>=8.3")
        if any(t["kind"] == "text" and t.get("resource") == "gpu" for t in self.cfg["tasks"]):
            packages.extend(["transformers>=4.46,<6", "safetensors>=0.4", "sentencepiece>=0.2"])
        if any(t["kind"] == "tabular" and t.get("resource") == "gpu" for t in self.cfg["tasks"]):
            packages.append("catboost>=1.2")
        packages.extend(self.options.get("extra_packages", []))
        code = "import subprocess,sys; subprocess.check_call([sys.executable,'-m','pip','install','--disable-pip-version-check',*sys.argv[1:]])"
        with ThreadPoolExecutor(max_workers=2) as pool:
            install = pool.submit(self.ssh, [self.remote_python, "-c", code, *packages], deadline=setup_end,
                                  log=self.run_dir / "remote-install.log")
            upload = pool.submit(self.copy, data, f"root@{self.host}:{self.remote_root}/data.tar", deadline=setup_end)
            install.result()
            upload.result()
        self.ssh(["env", "PYTHONPATH=" + self.remote_root, self.remote_python, "-c",
                  "from race90.bundle import safe_extract_tar; import sys; safe_extract_tar(sys.argv[1],sys.argv[2])",
                  self.remote_root + "/data.tar", self.remote_root], deadline=setup_end)
        self.ssh(["rm", "-f", self.remote_root + "/data.tar"], deadline=setup_end)
        data.unlink(missing_ok=True)

    def run_task(self, task, gpu, stage_deadline, threads):
        output = self.run_dir / task["name"]
        cfg = dict(self.cfg)
        cfg["data_root"], cfg["project_root"] = self.remote_root + "/data", self.remote_root
        cfg["runpod"] = {"enabled": False}
        cfg["dependency_outputs"] = {}
        with self.dependency_lock:
            for parent in task.get("depends_on", []):
                if parent not in self.uploaded_dependencies:
                    selected = Path(self.cfg["dependency_outputs"][parent])
                    dep_tar = self.run_dir / f"dependency-{parent}.tar"
                    with tarfile.open(dep_tar, "w") as archive:
                        for p in selected.iterdir():
                            if p.is_file():
                                archive.add(p, arcname=p.name, recursive=False)
                    remote_dep = self.remote_root + "/dependencies/" + parent
                    self.ssh(["mkdir", "-p", remote_dep], deadline=stage_deadline)
                    self.copy(dep_tar, f"root@{self.host}:{remote_dep}/artifacts.tar", deadline=stage_deadline)
                    self.ssh(["env", "PYTHONPATH=" + self.remote_root, self.remote_python, "-c",
                              "from race90.bundle import safe_extract_tar; import sys; safe_extract_tar(sys.argv[1],sys.argv[2])",
                              remote_dep + "/artifacts.tar", remote_dep], deadline=stage_deadline)
                    dep_tar.unlink(missing_ok=True)
                    self.uploaded_dependencies[parent] = remote_dep
                cfg["dependency_outputs"][parent] = self.uploaded_dependencies[parent]
        remote_output = self.remote_root + "/results/" + task["name"]
        payload = {"cfg": cfg, "task": task, "output": remote_output, "stage": "upgrade",
                   "deadline": stage_deadline, "threads": threads, "device": "cuda:0"}
        local_payload = output / "remote-payload.json"
        atomic_json(local_payload, payload)
        remote_payload = self.remote_root + f"/payload-{task['name']}.json"
        self.copy(local_payload, f"root@{self.host}:{remote_payload}", deadline=stage_deadline)
        # Remote process self-enforces its timeout even if SSH disconnects. Client
        # allows a little extra time to transfer already committed candidates.
        seconds = max(1, int(stage_deadline - time.time() - 10))
        command = ["env", "PYTHONPATH=" + self.remote_root, "CUDA_VISIBLE_DEVICES=" + str(gpu),
                   "OMP_NUM_THREADS=" + str(threads), "OPENBLAS_NUM_THREADS=" + str(threads),
                   "timeout", "--signal=TERM", "--kill-after=3s", str(seconds),
                   self.remote_python, "-m", "race90", "_worker", remote_payload]
        error = None
        try:
            self.ssh(command, deadline=stage_deadline, log=output / "upgrade.log")
        except (RuntimeError, TimeoutError) as e:
            error = e
        # Retrieval is independent of training success. An interrupted later
        # epoch must not hide earlier completed checkpoints.
        limit = self.cfg.get("_timeline", {}).get("retrieval_deadline", self.deadline + 90)
        retrieval_end = min(limit, time.time() + 90)
        archive_name = self.remote_root + f"/result-{task['name']}.tar"
        self.ssh(["env", "PYTHONPATH=" + self.remote_root, self.remote_python, "-c",
                  "from race90.remote import pack_result; import sys; pack_result(sys.argv[1], sys.argv[2])",
                  remote_payload, archive_name],
                 deadline=retrieval_end, log=output / "retrieve.log", cancellable=False)
        local_archive = output / "remote-results.tar"
        self.copy(f"root@{self.host}:{archive_name}", local_archive, deadline=retrieval_end, cancellable=False)
        safe_extract_tar(local_archive, self.run_dir)
        local_archive.unlink(missing_ok=True)
        if error is not None:
            raise error

    def cleanup(self):
        with self.lock:
            if self.closed or not self.pod_id:
                return
            if not self.owned:
                self.record(status="attached_pod_retained")
                self.closed = True
                return
            try:
                cleanup_end = min(time.time() + 25, self.cfg.get("_timeline", {}).get("retrieval_deadline", float("inf")))
                run_command([self.cli, "pod", "delete", self.pod_id], cleanup_end, capture=True)
                (self.run_dir / "lease-cancelled").touch()
                self.record(status="deleted")
                self.closed = True
            except Exception as e:
                self.record(status="cleanup_failed", error=str(e))


def pack_result(payload_path, archive_path):
    """Return only the best committed checkpoint, not all epochs/optimizer runs."""
    from .selection import select_best
    payload = json.loads(Path(payload_path).read_text())
    output = Path(payload["output"])
    candidate, _ = select_best(payload["cfg"], payload["task"], output)
    with tarfile.open(archive_path, "w") as archive:
        archive.add(candidate, arcname=str(candidate.relative_to(output.parent)))
        for name in ("audit.json", "upgrade-status.json"):
            path = output / name
            if path.is_file():
                archive.add(path, arcname=str(path.relative_to(output.parent)))


def lease_watchdog(cli, pod_id, end, cancelled):
    while time.time() < float(end):
        if Path(cancelled).exists():
            return 0
        time.sleep(min(5, max(.01, float(end) - time.time())))
    for attempt in range(3):
        try:
            run_command([cli, "pod", "delete", pod_id], time.time() + 25, capture=True)
            return 0
        except Exception as e:
            print(f"lease cleanup attempt {attempt + 1}: {e}", flush=True)
            time.sleep(3)
    return 1
