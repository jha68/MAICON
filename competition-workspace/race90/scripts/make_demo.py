"""Build three synthetic problems. No Runpod credentials or GPU required."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import yaml
import nbformat
from PIL import Image


def make_demo(root):
    root = Path(root).resolve()
    data = root / "data"
    rng = np.random.default_rng(42)
    for name in ("q1", "q2", "q3"):
        folder = data / name
        folder.mkdir(parents=True, exist_ok=True)
        nb = nbformat.v4.new_notebook(cells=[nbformat.v4.new_markdown_cell(f"# {name} 합성 문제\nCSV를 제출하세요."),
                                            nbformat.v4.new_code_cell("# 원본 미완성 셀: 내보내기에서 실행하지 않습니다.\nraise RuntimeError('starter cell')")])
        nbformat.write(nb, folder / "problem.ipynb")
    rows = []
    for i in range(60):
        label = i % 2
        for step in range(12):
            rows.append({"sample_id": i, "subject_id": i // 6, "timestamp": step,
                         "acc_x": label * 3 + rng.normal(0, .1), "acc_y": rng.normal(),
                         "acc_z": label + rng.normal(0, .1), "label": label})
    train = pd.DataFrame(rows)
    test = train[train.sample_id < 10].copy()
    test["sample_id"] += 1000
    test = test.drop(columns="label")
    train.to_csv(data / "q1/train.csv", index=False)
    test.to_csv(data / "q1/test.csv", index=False)
    pd.DataFrame({"sample_id": sorted(test.sample_id.unique(), reverse=True), "label": 0}).to_csv(data / "q1/sample_submission.csv", index=False)
    images = data / "q2/images"
    images.mkdir()
    frames = []
    for start, count, labeled in ((0, 40, True), (1000, 8, False)):
        records = []
        for i in range(count):
            color = [220, 20, 20] if i % 2 == 0 else [20, 220, 20]
            arr = np.clip(np.array(color) + rng.normal(0, 4, (24, 24, 3)), 0, 255).astype("uint8")
            filename = f"{start+i}.png"
            Image.fromarray(arr).save(images / filename)
            records.append({"id": start + i, "filename": filename, **({"label": i % 2} if labeled else {})})
        frames.append(pd.DataFrame(records))
    frames[0].to_csv(data / "q2/train.csv", index=False)
    frames[1].to_csv(data / "q2/test.csv", index=False)
    pd.DataFrame({"id": frames[1].id.iloc[::-1].to_numpy(), "label": 0}).to_csv(data / "q2/sample_submission.csv", index=False)
    records = [{"id": i, "text": ("좋아요 재미있고 만족합니다" if i % 2 == 0 else "별로예요 지루하고 실망합니다") + f" 후기 {i}", "label": i % 2}
               for i in range(60)]
    train = pd.DataFrame(records)
    test = train.iloc[:10].drop(columns="label").copy()
    test["id"] += 1000
    train.to_csv(data / "q3/train.csv", index=False)
    test.to_csv(data / "q3/test.csv", index=False)
    pd.DataFrame({"id": test.id.iloc[::-1].to_numpy(), "label": 0}).to_csv(data / "q3/sample_submission.csv", index=False)
    cfg = yaml.safe_load((Path(__file__).parents[1] / "config.example.yaml").read_text())
    cfg.update(total_minutes=2, finalize_minutes=.5, cpu_threads=3, time_management={},
               data_root=str(data), output_root=str(root / "runs"))
    cfg["runpod"]["enabled"] = False
    for task in cfg["tasks"]:
        task["improve"] = False
        task["baseline_seconds"] = 30
    config = root / "demo.yaml"
    config.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="demo")
    args = parser.parse_args()
    print(make_demo(args.output))
