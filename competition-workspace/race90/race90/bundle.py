from pathlib import Path
import io
import tarfile
import zipfile
import importlib.util


def source_files(cfg):
    package = Path(__file__).parent
    files = {"race90/" + p.name: p for p in package.glob("*.py")}
    for task in cfg["tasks"]:
        for key in ("adapter", "exporter"):
            if not task.get(key):
                continue
            module = task[key].split(":", 1)[0].split(".")[0]
            root = Path(cfg["project_root"])
            folder, single = root / module, root / (module + ".py")
            if folder.is_dir():
                for p in folder.rglob("*.py"):
                    files[str(p.relative_to(root))] = p
            elif single.is_file():
                files[single.name] = single
            elif module != "race90":
                raise ValueError(f"Place custom module {module} inside project_root before bundling")
    return files


def source_zip(cfg):
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as z:
        for name, path in source_files(cfg).items():
            z.write(path, name)
    return output.getvalue()


def safe_extract_tar(path, dest):
    dest = Path(dest).resolve()
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path) as archive:
        for member in archive.getmembers():
            target = (dest / member.name).resolve()
            if not target.is_relative_to(dest) or not (member.isfile() or member.isdir()):
                raise ValueError(f"Unsafe archive entry: {member.name}")
        try:
            archive.extractall(dest, filter="data")
        except TypeError:  # Python 3.10 without backported tar filters.
            archive.extractall(dest)  # Entries and link types were validated above.
