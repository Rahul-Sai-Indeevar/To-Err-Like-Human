"""Extract only evaluation artifacts from a saved Kaggle workspace, without overwriting source."""

import argparse, pathlib, shutil, zipfile


def main():
    p = argparse.ArgumentParser()
    p.add_argument("archive")
    p.add_argument("--dest", default="data/kaggle_full_8epoch")
    a = p.parse_args()
    dest = pathlib.Path(a.dest).resolve()
    dest.mkdir(parents=True, exist_ok=True)
    chosen = {f"{m}_seed{s}" for m in ["ce", "graph_a0p3"] for s in [0, 1, 2]}
    count = 0
    with zipfile.ZipFile(a.archive) as z:
        for item in z.infolist():
            rel = pathlib.PurePosixPath(item.filename)
            if rel.is_absolute() or ".." in rel.parts:
                raise ValueError("Unsafe archive path")
            name = item.filename
            keep = name.startswith("data/emoset_full/") and (
                "/images/test/" in name or "/images/" not in name
            )
            if name.startswith("emoset_graph_alpha_sweep_2gpu/"):
                keep = not name.endswith(".pt") or (
                    "/runs/" in name
                    and name.split("/runs/")[1].split("/")[0] in chosen
                    and name.endswith("/best.pt")
                )
            if not keep or item.is_dir():
                continue
            target = dest.joinpath(*rel.parts)
            if not target.is_relative_to(dest):
                raise ValueError("Archive path escaped destination")
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                if target.stat().st_size != item.file_size:
                    raise ValueError(f"Existing artifact has wrong size: {target}")
                continue
            temp = target.with_name(target.name + ".part")
            with z.open(item) as src, temp.open("wb") as out:
                shutil.copyfileobj(src, out)
            temp.replace(target)
            count += 1
    print(
        "Imported",
        count,
        "artifacts; full training images and optimizer checkpoints remain in source ZIP.",
    )


if __name__ == "__main__":
    main()
