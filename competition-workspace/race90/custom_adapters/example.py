"""Copy this adapter and change it to match the actual problem statement."""
from pathlib import Path
import pandas as pd
from race90.config import resolve_data
from race90.io import make_submission, read_csv


def solve(ctx):
    """CPU baseline or GPU upgrade; ctx.stage tells you which phase is running."""
    # Parent directories contain the immutable chosen model and published artifacts.
    for name, directory in ctx.dependencies.items():
        print(f"Dependency {name}: {directory}", flush=True)
    test = read_csv(ctx.cfg, ctx.task["test_csv"], ctx.task)
    # Replace with actual validation, training and inference. The external runner
    # enforces a hard deadline, but polling ctx.seconds_left() permits clean saves.
    raise NotImplementedError("Implement this problem's modeling and CSV schema before running")
    # frame = make_submission(ctx.cfg, ctx.task, test[ctx.task["id_column"]], predictions)
    # ctx.publish(frame, validation_score, {"algorithm": "my-model", "feature_mode": "custom"},
    #             artifacts={"train_features.csv": train_oof_path,
    #                        "test_features.csv": test_features_path,
    #                        "model.pt": checkpoint_path})


def export_submission(cfg, task, test, results):
    """YOLO exporter: return a DataFrame exactly matching sample_submission."""
    raise NotImplementedError("Implement the exact detection output format required by the question")
