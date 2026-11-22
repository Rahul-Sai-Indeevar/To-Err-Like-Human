"""Package auditable wavelet evidence without raw photographs, caches or weights."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile


def digest_file(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def package(root, study, destination):
    root, study, destination = root.resolve(), study.resolve(), destination.resolve()
    completion = json.loads((study / "automatic_completion.json").read_text())
    if completion["status"] != "complete":
        raise ValueError("Package only completed experiments")
    for name in ("verification.json", "diagnostics_verification.json", "input_verification.json"):
        if not (study / name).is_file():
            raise ValueError(f"Required audit missing: {name}")
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=root).decode().split("\0")
    files = {name.replace("\\", "/"): root / name for name in tracked if name and (root / name).is_file()}
    forbidden = {".pt", ".pth", ".npy", ".jpg", ".jpeg", ".webp", ".tar", ".gz", ".zip"}
    for path in study.rglob("*"):
        if path.is_file() and path.suffix.lower() not in forbidden and not path.name.endswith(".part"):
            files[path.relative_to(root).as_posix()] = path
    for protocol_path in study.glob("*/protocol.json"):
        protocol = json.loads(protocol_path.read_text())
        data = root / protocol["args"]["data_root"]
        for name in ("classes.txt", "train.csv", "val.csv", "test.csv", "manifest.json"):
            path = data / name
            files[path.relative_to(root).as_posix()] = path
    if any(path.suffix.lower() in {".pt", ".pth", ".npy"} for path in files.values()):
        raise ValueError("Weights or caches unexpectedly present in tracked/evidence files")
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(),
                "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
                "branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=root, text=True).strip(),
                "scope": "Code, report, per-image probabilities, histories, protocols, source snapshots and split metadata; raw photographs, caches, weights excluded",
                "files": [{"path": name, "bytes": path.stat().st_size, "sha256": digest_file(path)} for name, path in sorted(files.items())]}
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_suffix(".part.zip")
    with zipfile.ZipFile(temp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, path in sorted(files.items()):
            archive.write(path, name)
        archive.writestr("PACKAGE_MANIFEST.json", json.dumps(manifest, indent=2))
    with zipfile.ZipFile(temp) as archive:
        if archive.testzip() is not None:
            raise ValueError("ZIP CRC verification failed")
        for entry in manifest["files"]:
            content = archive.read(entry["path"])
            if len(content) != entry["bytes"] or hashlib.sha256(content).hexdigest() != entry["sha256"]:
                raise ValueError(f"Package SHA verification failed: {entry['path']}")
    temp.replace(destination)
    sha = digest_file(destination)
    destination.with_suffix(".sha256.txt").write_text(f"{sha}  {destination.name}\n", encoding="utf8")
    print(json.dumps({"package": str(destination), "bytes": destination.stat().st_size,
                      "entries": len(manifest["files"]) + 1, "sha256": sha, "all_entry_hashes_verified": True}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    repo = Path(__file__).resolve().parents[1]
    parser.add_argument("--study", type=Path, default=repo / "outputs/wavelet_study")
    parser.add_argument("--out", type=Path, default=repo / "outputs/wavelet_evidence_bundle.zip")
    args = parser.parse_args()
    package(repo, args.study, args.out)
