"""GPU sensor CNN and text transfer learning with CPU-compatible checkpoints."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from .data import load_dataset, recording_times
from .io import read_csv, atomic_json
from .models import split_indices, metric_score, MINIMIZE
from .vision import Probabilities, output_probs


def sequence_tensor(frame, task):
    sensors, ident = task["sensor_columns"], task["id_column"]
    if task.get("time_column"):
        frame = frame.sort_values([ident, task["time_column"]], kind="stable")
    length = int(task.get("sequence_length", 256))
    ids, sequences, metadata = [], [], []
    for key, group in frame.groupby(ident, sort=False):
        values = group[sensors].to_numpy(dtype=np.float32)
        channels = []
        for col in values.T:
            valid = np.isfinite(col)
            if not valid.any():
                channels.append(np.zeros(length, dtype=np.float32))
            else:
                channels.append(np.interp(np.linspace(0, max(0, len(col) - 1), length),
                                          np.arange(len(col))[valid], col[valid]).astype(np.float32))
        sequences.append(channels)
        ids.append(key)
        metadata.append(group.iloc[0])
    return np.array(sequences, dtype=np.float32), pd.Series(ids), pd.DataFrame(metadata).reset_index(drop=True)


def sensor_net(channels, classes):
    import torch
    return torch.nn.Sequential(torch.nn.Conv1d(channels, 64, 5, padding=2), torch.nn.ReLU(),
                               torch.nn.Conv1d(64, 128, 5, padding=2), torch.nn.ReLU(),
                               torch.nn.AdaptiveAvgPool1d(1), torch.nn.Flatten(),
                               torch.nn.Linear(128, classes))


def tensor_probs(model, tensor, device, batch_size):
    import torch
    model.eval()
    output = []
    with torch.inference_mode():
        for start in range(0, len(tensor), batch_size):
            output.append(model(torch.as_tensor(tensor[start:start + batch_size], device=device)).softmax(1).cpu().numpy())
    return np.concatenate(output)


def train_sensor(ctx):
    import torch
    torch.set_num_threads(ctx.threads)
    torch.manual_seed(ctx.cfg["seed"])
    task = ctx.task
    if task["objective"] != "classification":
        raise ValueError("Built-in sensor CNN supports classification; use CPU or custom for regression")
    train = read_csv(ctx.cfg, task["train_csv"], task)
    X, _, metadata = sequence_tensor(train, task)
    metadata = recording_times(train, metadata, task)
    Xt, ids, _ = sequence_tensor(read_csv(ctx.cfg, task["test_csv"], task), task)
    y = metadata[task["target"]]
    classes = sorted(y.unique().tolist())
    labels = np.array([classes.index(v) for v in y])
    a, b, split_hash = split_indices(task, y, metadata, ctx.cfg["seed"])
    mean, std = X[a].mean(axis=(0, 2), keepdims=True), X[a].std(axis=(0, 2), keepdims=True).clip(1e-6)
    X, Xt = (X - mean) / std, (Xt - mean) / std
    model = sensor_net(len(task["sensor_columns"]), len(classes)).to(ctx.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(task.get("learning_rate", 1e-3)))
    batch = int(task.get("batch_size", 128))
    rng = np.random.default_rng(ctx.cfg["seed"])
    best, stale = None, 0
    checkpoint = ctx.output / "sensor-current.pt"
    for epoch in range(int(task.get("epochs", 30))):
        if ctx.seconds_left() < task.get("inference_reserve_seconds", 60):
            break
        order = rng.permutation(a)
        model.train()
        for start in range(0, len(order), batch):
            if ctx.seconds_left() < task.get("inference_reserve_seconds", 60):
                break
            idx = order[start:start + batch]
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(model(torch.as_tensor(X[idx], device=ctx.device)),
                                                      torch.as_tensor(labels[idx], device=ctx.device))
            loss.backward()
            optimizer.step()
        probs = tensor_probs(model, X[b], ctx.device, batch)
        score = metric_score(Probabilities(probs, classes), None, y.iloc[b], task)
        improved = best is None or (score < best if task["metric"] in MINIMIZE else score > best)
        if not improved:
            stale += 1
            if stale >= int(task.get("patience", 5)):
                break
            continue
        stale = 0
        frame = output_probs(ctx.cfg, task, ids, tensor_probs(model, Xt, ctx.device, batch), classes)
        torch.save({"state_dict": model.state_dict(), "classes": classes,
                    "mean": torch.tensor(mean), "std": torch.tensor(std)}, checkpoint)
        ctx.publish(frame, score, {"algorithm": "sensor_cnn", "feature_mode": "sensor_cnn", "epoch": epoch + 1,
                                  "split_hash": split_hash, "refit_all_rows": False}, artifacts={"model.pt": checkpoint})
        best = score
    if best is None:
        raise RuntimeError("No sensor CNN checkpoint completed within budget")


def text_batches(tokenizer, texts, indices, device, task):
    batch = int(task.get("batch_size", 32))
    for start in range(0, len(indices), batch):
        idx = indices[start:start + batch]
        encoded = tokenizer(texts.iloc[idx].tolist(), padding=True, truncation=True,
                            max_length=int(task.get("max_length", 128)), return_tensors="pt")
        yield idx, {k: v.to(device) for k, v in encoded.items()}


def text_probs(model, tokenizer, texts, device, task):
    import torch
    model.eval()
    out = []
    with torch.inference_mode():
        for _, batch in text_batches(tokenizer, texts, np.arange(len(texts)), device, task):
            out.append(model(**batch).logits.softmax(1).cpu().numpy())
    return np.concatenate(out)


def train_text(ctx):
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    torch.set_num_threads(ctx.threads)
    torch.manual_seed(ctx.cfg["seed"])
    task = ctx.task
    if task["objective"] != "classification":
        raise ValueError("Built-in transformer supports classification; use CPU or custom for regression")
    X, Xt, y, ids, metadata = load_dataset(ctx.cfg, task)
    a, b, split_hash = split_indices(task, y, metadata, ctx.cfg["seed"])
    classes = sorted(y.unique().tolist())
    labels = np.array([classes.index(v) for v in y])
    model_name = task.get("model_name", "klue/roberta-small")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(classes),
                                                               ignore_mismatched_sizes=True).to(ctx.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(task.get("learning_rate", 2e-5)))
    amp = ctx.device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    rng = np.random.default_rng(ctx.cfg["seed"])
    best = None
    current = ctx.output / "transformer-current"
    current.mkdir(exist_ok=True)
    for epoch in range(int(task.get("epochs", 3))):
        if ctx.seconds_left() < task.get("inference_reserve_seconds", 120):
            break
        model.train()
        for idx, encoded in text_batches(tokenizer, X, rng.permutation(a), ctx.device, task):
            if ctx.seconds_left() < task.get("inference_reserve_seconds", 120):
                break
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda" if amp else "cpu", enabled=amp):
                loss = model(**encoded, labels=torch.as_tensor(labels[idx], device=ctx.device)).loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        score = metric_score(Probabilities(text_probs(model, tokenizer, X.iloc[b].reset_index(drop=True), ctx.device, task), classes),
                             None, y.iloc[b], task)
        if best is not None and (score >= best if task["metric"] in MINIMIZE else score <= best):
            continue
        frame = output_probs(ctx.cfg, task, ids, text_probs(model, tokenizer, Xt, ctx.device, task), classes)
        # Save a compact FP16 state without mutating the training model/optimizer.
        half_state = {k: v.detach().cpu().half() if v.is_floating_point() else v.detach().cpu()
                      for k, v in model.state_dict().items()}
        model.save_pretrained(current, state_dict=half_state, safe_serialization=True)
        tokenizer.save_pretrained(current)
        atomic_json(current / "classes.json", classes)
        ctx.publish(frame, score, {"algorithm": "transformer", "feature_mode": "transformer", "epoch": epoch + 1,
                                  "model_name": model_name, "split_hash": split_hash, "refit_all_rows": False},
                    artifacts={p.name: p for p in current.iterdir() if p.is_file()})
        best = score
    if best is None:
        raise RuntimeError("No transformer checkpoint completed within budget")


def predict_neural_checkpoint(cfg, task, candidate, device):
    import torch
    if task["kind"] == "timeseries":
        checkpoint = torch.load(candidate / "model.pt", map_location="cpu", weights_only=True)
        X, ids, _ = sequence_tensor(read_csv(cfg, task["test_csv"], task), task)
        X = (X - checkpoint["mean"].numpy()) / checkpoint["std"].numpy()
        classes = checkpoint["classes"]
        model = sensor_net(len(task["sensor_columns"]), len(classes))
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)
        probs = tensor_probs(model, X, device, int(task.get("batch_size", 128)))
    else:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        tokenizer = AutoTokenizer.from_pretrained(candidate, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(candidate, local_files_only=True).float().to(device)
        test = read_csv(cfg, task["test_csv"], task)
        X = test[task["text_columns"]].fillna("").astype(str).agg(" ".join, axis=1)
        ids = test[task["id_column"]]
        classes = json.loads((candidate / "classes.json").read_text())
        probs = text_probs(model, tokenizer, X, device, task)
    return output_probs(cfg, task, ids, probs, classes)
