from pathlib import Path
import time
import numpy as np
import pandas as pd
from .config import resolve_data
from .data import image_path
from .io import read_csv, make_submission
from .models import split_indices, metric_score


def normalized_yolo_yaml(cfg, task, output):
    """Make YAML/txt dataset paths portable between local files and the Pod."""
    import yaml
    source = resolve_data(cfg, task["data_yaml"])
    data = yaml.safe_load(source.read_text())
    root_spec = Path(data.get("path", "."))
    if root_spec.is_absolute():
        raise ValueError("YOLO YAML path must be relative to its YAML file for remote portability")
    root = (source.parent / root_spec).resolve()
    data_root = Path(cfg["data_root"]).resolve()
    if not root.is_relative_to(data_root):
        raise ValueError("YOLO dataset escapes data_root")
    split_images = {}
    for split in ("train", "val"):
        entries = data.get(split)
        if not entries:
            raise ValueError(f"YOLO dataset requires an explicit {split} split")
        entries = entries if isinstance(entries, list) else [entries]
        resolved, images = [], []
        for entry in entries:
            relative = Path(entry)
            if relative.is_absolute():
                raise ValueError("YOLO split paths must be relative to the dataset path")
            path = (root / relative).resolve()
            if not path.is_relative_to(data_root) or not path.exists():
                raise ValueError(f"Invalid YOLO {split} path: {entry}")
            if path.is_dir():
                images.extend(str(p.resolve()) for p in path.rglob("*")
                              if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"})
                resolved.append(str(path))
            elif path.suffix == ".txt":
                listed = []
                for line in path.read_text().splitlines():
                    if not line.strip():
                        continue
                    item = Path(line.strip())
                    if item.is_absolute():
                        raise ValueError("YOLO image lists must use paths relative to the txt file")
                    item = (path.parent / item).resolve()
                    if not item.is_relative_to(data_root) or not item.is_file():
                        raise ValueError(f"Invalid image list entry: {line}")
                    listed.append(str(item))
                listing = Path(output) / f"{split}-{len(resolved)}.runtime.txt"
                listing.parent.mkdir(parents=True, exist_ok=True)
                listing.write_text("\n".join(listed) + "\n")
                resolved.append(str(listing))
                images.extend(listed)
            else:
                raise ValueError("YOLO train/val must be an image directory or txt image list")
        if not images or len(set(images)) != len(images):
            raise ValueError(f"YOLO {split} split is empty or contains duplicate images")
        split_images[split] = set(images)
        data[split] = resolved if len(resolved) > 1 else resolved[0]
    if split_images["train"] & split_images["val"]:
        raise ValueError("YOLO training and validation images overlap")
    data["path"] = str(root)
    runtime = Path(output) / "dataset.runtime.yaml"
    runtime.parent.mkdir(parents=True, exist_ok=True)
    runtime.write_text(yaml.safe_dump(data, sort_keys=False))
    return runtime, {split: len(images) for split, images in split_images.items()}


def torch_components():
    import torch
    from torchvision import models, transforms
    return torch, models, transforms


def image_loader(cfg, task, frame, labels=None, train=False, shuffle=False):
    torch, _, T = torch_components()
    from PIL import Image
    size = int(task.get("image_size", 224))
    ops = ([T.RandomResizedCrop(size, scale=(0.7, 1.0)), T.RandomHorizontalFlip()]
           if train else [T.Resize(size + 32), T.CenterCrop(size)])
    transform = T.Compose(ops + [T.ToTensor(), T.Normalize([.485, .456, .406], [.229, .224, .225])])

    class Images(torch.utils.data.Dataset):
        def __len__(self):
            return len(frame)

        def __getitem__(self, idx):
            with Image.open(image_path(cfg, task, frame.iloc[idx][task["image_column"]])) as im:
                x = transform(im.convert("RGB"))
            return x, (int(labels[idx]) if labels is not None else 0)

    # Dataset is local to this function; workers=0 also avoids extra CPU contention.
    return torch.utils.data.DataLoader(Images(), batch_size=int(task.get("batch_size", 32)),
                                      shuffle=shuffle, num_workers=0, pin_memory=torch.cuda.is_available())


def image_probs(net, loader, device):
    torch, _, _ = torch_components()
    net.eval()
    output = []
    with torch.inference_mode():
        for images, _ in loader:
            output.append(net(images.to(device)).softmax(1).cpu().numpy())
    return np.concatenate(output)


class Probabilities:
    def __init__(self, probs, classes):
        self.probs, self.classes_ = probs, np.asarray(classes)

    def predict(self, _):
        return self.classes_[self.probs.argmax(axis=1)]

    def predict_proba(self, _):
        return self.probs


def output_probs(cfg, task, ids, probs, classes):
    from .models import predictions
    return make_submission(cfg, task, ids, predictions(Probabilities(probs, classes), None, task))


def train_resnet(ctx):
    torch, models, _ = torch_components()
    torch.set_num_threads(ctx.threads)
    torch.manual_seed(ctx.cfg["seed"])
    np.random.seed(ctx.cfg["seed"])
    if ctx.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; baseline is retained")
    task = dict(ctx.task)
    train = read_csv(ctx.cfg, task["train_csv"], task)
    test = read_csv(ctx.cfg, task["test_csv"], task)
    y = train[task["target"]]
    classes = sorted(y.unique().tolist())
    if task["objective"] != "classification":
        raise ValueError("Built-in image adapter supports classification; use custom for regression")
    a, b, split_hash = split_indices(task, y, train, ctx.cfg["seed"])
    labels = np.array([classes.index(v) for v in y])
    weights = models.ResNet18_Weights.DEFAULT if task.get("pretrained", True) else None
    net = models.resnet18(weights=weights)
    frozen = task.get("freeze_backbone", True)
    if frozen:
        for p in net.parameters():
            p.requires_grad = False
    net.fc = torch.nn.Linear(net.fc.in_features, len(classes))
    net.to(ctx.device)
    optimizer = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad],
                                 lr=float(task.get("learning_rate", 1e-3)))
    use_amp = ctx.device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best = None
    checkpoint = ctx.output / "resnet-current.pt"
    for epoch in range(int(task.get("epochs", 8))):
        if ctx.seconds_left() < task.get("inference_reserve_seconds", 90):
            break
        loader = image_loader(ctx.cfg, task, train.iloc[a].reset_index(drop=True), labels[a], train=True, shuffle=True)
        net.train()
        if frozen:
            net.eval()  # Keep frozen BatchNorm statistics unchanged.
            net.fc.train()
        try:
            for images, targets in loader:
                if ctx.seconds_left() < task.get("inference_reserve_seconds", 90):
                    break
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda" if use_amp else "cpu", enabled=use_amp):
                    loss = torch.nn.functional.cross_entropy(net(images.to(ctx.device)), targets.to(ctx.device))
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        except torch.cuda.OutOfMemoryError:
            if task.get("batch_size", 32) <= 1:
                raise
            task["batch_size"] = max(1, task.get("batch_size", 32) // 2)
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            continue
        val_loader = image_loader(ctx.cfg, task, train.iloc[b].reset_index(drop=True))
        probs = image_probs(net, val_loader, ctx.device)
        score = metric_score(Probabilities(probs, classes), None, y.iloc[b], task)
        from .models import MINIMIZE
        if best is not None and (score >= best if task["metric"] in MINIMIZE else score <= best):
            continue
        test_probs = image_probs(net, image_loader(ctx.cfg, task, test), ctx.device)
        frame = output_probs(ctx.cfg, task, test[task["id_column"]], test_probs, classes)
        torch.save({"state_dict": net.state_dict(), "classes": classes}, checkpoint)
        ctx.publish(frame, score, {"algorithm": "resnet18", "feature_mode": "resnet", "epoch": epoch + 1,
                                  "split_hash": split_hash, "train_rows": len(a), "valid_rows": len(b),
                                  "pretrained": bool(weights), "refit_all_rows": False},
                    artifacts={"model.pt": checkpoint})
        best = score
    if best is None:
        raise RuntimeError("No vision epoch completed within the deadline")


def yolo_predictions(cfg, task, weights, device):
    from ultralytics import YOLO
    from .worker import callable_from
    test = read_csv(cfg, task["test_csv"], task)
    paths = [str(image_path(cfg, task, v)) for v in test[task["image_column"]]]
    model = YOLO(str(weights))
    results = model.predict(paths, device=device, imgsz=int(task.get("image_size", 640)),
                            batch=int(task.get("batch_size", 8)), stream=True,
                            conf=float(task.get("confidence", 0.25)), verbose=False)
    if task.get("exporter"):
        return callable_from(task["exporter"])(cfg, task, test, results)
    # Explicit built-in format: one row per image, space-separated
    # class confidence xmin ymin xmax ymax blocks, absolute pixel coordinates.
    if task.get("detection_format") != "class_conf_xyxy":
        raise ValueError("Set detection_format=class_conf_xyxy or provide exporter=module:function")
    strings = []
    for result in results:
        boxes = result.boxes
        tokens = []
        if boxes is not None:
            for cls, conf, xyxy in zip(boxes.cls.cpu().tolist(), boxes.conf.cpu().tolist(), boxes.xyxy.cpu().tolist()):
                tokens.extend([str(int(cls)), f"{conf:.6f}", *(f"{v:.2f}" for v in xyxy)])
        strings.append(" ".join(tokens))
    return make_submission(cfg, task, test[task["id_column"]], np.array(strings, dtype=object))


def train_yolo(ctx):
    from ultralytics import YOLO
    task = ctx.task
    model = YOLO(task.get("weights", "yolo11n.pt"))
    best = [None]
    def publish_epoch(trainer):
        # on_model_save runs after checkpoint creation and metric calculation.
        if ctx.seconds_left() < task.get("inference_reserve_seconds", 90):
            trainer.stop = True
            return
        metric_key = "metrics/mAP50(B)" if task["metric"] == "map50" else "metrics/mAP50-95(B)"
        score = trainer.metrics.get(metric_key)
        if score is None or (best[0] is not None and score <= best[0]) or not Path(trainer.last).exists():
            return
        # Current epoch metrics belong to last.pt; best.pt can be from a different
        # epoch when the competition metric differs from YOLO's own fitness.
        frame = yolo_predictions(ctx.cfg, task, trainer.last, ctx.device)
        ctx.publish(frame, float(score), {"algorithm": "YOLO", "feature_mode": "yolo", "epoch": trainer.epoch + 1,
                                          "validation": task["data_yaml"], "refit_all_rows": False},
                    artifacts={"model.pt": trainer.last})
        best[0] = float(score)
    model.add_callback("on_model_save", publish_epoch)
    train_seconds = max(1, ctx.seconds_left() - task.get("inference_reserve_seconds", 90))
    dataset_yaml, _ = normalized_yolo_yaml(ctx.cfg, task, ctx.output)
    model.train(data=str(dataset_yaml), device=ctx.device,
                epochs=int(task.get("epochs", 30)), time=train_seconds / 3600,
                imgsz=int(task.get("image_size", 640)), batch=int(task.get("batch_size", 8)),
                workers=0, seed=ctx.cfg["seed"], patience=5, save=True,
                project=str(ctx.output / "yolo_training"), name="train", exist_ok=True, verbose=False)
    if best[0] is None:
        raise RuntimeError("YOLO produced no validated checkpoint; fallback retained")


def train_vision(ctx):
    if ctx.task["kind"] == "yolo":
        train_yolo(ctx)
    else:
        train_resnet(ctx)


def resnet_embeddings(cfg, task, frame, net, device="cpu"):
    """One forward pass per image; the backbone is never trained on exam data."""
    torch, _, _ = torch_components()
    net.eval()
    rows = []
    with torch.inference_mode():
        for images, _ in image_loader(cfg, task, frame):
            rows.append(net(images.to(device)).cpu().numpy())
    return pd.DataFrame(np.concatenate(rows))


def train_image_embeddings(ctx):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from .models import fit_cpu_dataset
    torch, models, _ = torch_components()
    torch.set_num_threads(ctx.threads)
    torch.manual_seed(ctx.cfg["seed"])
    task = ctx.task
    if task["objective"] != "classification":
        raise ValueError("Image embeddings currently support classification")
    weights = models.ResNet18_Weights.DEFAULT if task.get("pretrained", True) else None
    net = models.resnet18(weights=weights)
    net.fc = torch.nn.Identity()
    train = read_csv(ctx.cfg, task["train_csv"], task)
    test = read_csv(ctx.cfg, task["test_csv"], task)
    X = resnet_embeddings(ctx.cfg, task, train, net)
    Xt = resnet_embeddings(ctx.cfg, task, test, net)
    checkpoint = ctx.output / "backbone-current.pt"
    torch.save(net.state_dict(), checkpoint)
    classifier = make_pipeline(StandardScaler(), LogisticRegression(C=task.get("C", 1.0), max_iter=300,
                                                                    random_state=ctx.cfg["seed"]))
    fit_cpu_dataset(ctx, X, Xt, train[task["target"]], test[task["id_column"]], train,
                    model=classifier, details={"feature_mode": "resnet_features", "pretrained": bool(weights)},
                    artifacts={"backbone.pt": checkpoint})


def predict_vision_checkpoint(cfg, task, candidate, device):
    if task["kind"] == "yolo":
        return yolo_predictions(cfg, task, candidate / "model.pt", device)
    torch, models, _ = torch_components()
    if (candidate / "backbone.pt").exists():
        import joblib
        from .models import predictions
        net = models.resnet18(weights=None)
        net.fc = torch.nn.Identity()
        net.load_state_dict(torch.load(candidate / "backbone.pt", map_location="cpu", weights_only=True))
        net.to(device)
        test = read_csv(cfg, task["test_csv"], task)
        X = resnet_embeddings(cfg, task, test, net, device)
        classifier = joblib.load(candidate / "model.joblib")
        return make_submission(cfg, task, test[task["id_column"]], predictions(classifier, X, task))
    checkpoint = torch.load(candidate / "model.pt", map_location="cpu", weights_only=True)
    net = models.resnet18(weights=None)
    net.fc = torch.nn.Linear(net.fc.in_features, len(checkpoint["classes"]))
    net.load_state_dict(checkpoint["state_dict"])
    net.to(device)
    test = read_csv(cfg, task["test_csv"], task)
    probs = image_probs(net, image_loader(cfg, task, test), device)
    return output_probs(cfg, task, test[task["id_column"]], probs, checkpoint["classes"])
