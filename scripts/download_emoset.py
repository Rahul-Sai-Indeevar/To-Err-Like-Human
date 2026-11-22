import urllib.request, json, pathlib, concurrent.futures, hashlib, time

root = pathlib.Path("data/emoset_source")
root.mkdir(parents=True, exist_ok=True)
repo = "udeman/EmoSet-118K"
previous = root / "provenance.json"
info = (
    json.loads(previous.read_text())
    if previous.exists()
    else json.load(
        urllib.request.urlopen("https://huggingface.co/api/datasets/" + repo)
    )
)
rev = info.get("revision", info.get("sha"))
paths = [
    "data/train-00000-of-00018.parquet",
    "data/train-00001-of-00018.parquet",
    "data/val-00000-of-00002.parquet",
    "data/test-00000-of-00004.parquet",
]
remote_files = json.load(
    urllib.request.urlopen(
        f"https://huggingface.co/api/datasets/{repo}/tree/{rev}/data"
    )
)
expected_hashes = {
    item["path"]: item.get("lfs", {}).get("oid") for item in remote_files
}


def get(path):
    out = root / pathlib.Path(path).name
    if not out.exists():
        url = (
            f"https://huggingface.co/datasets/{repo}/resolve/{rev}/{path}?download=true"
        )
        print("Downloading", path, flush=True)
        for attempt in range(4):
            try:
                with (
                    urllib.request.urlopen(url, timeout=120) as r,
                    open(str(out) + ".part", "wb") as f,
                ):
                    while b := r.read(4 * 1024 * 1024):
                        f.write(b)
                pathlib.Path(str(out) + ".part").replace(out)
                break
            except Exception as e:
                print(type(e).__name__, str(e)[:200], flush=True)
                if attempt == 3:
                    raise
                time.sleep(2)
    h = hashlib.sha256()
    with out.open("rb") as f:
        while b := f.read(8 * 1024 * 1024):
            h.update(b)
    if expected_hashes.get(path) != h.hexdigest():
        raise ValueError(f"SHA256 mismatch against pinned mirror for {path}")
    print("Ready", out.name, out.stat().st_size, flush=True)
    return {"path": path, "bytes": out.stat().st_size, "sha256": h.hexdigest()}


with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
    files = list(ex.map(get, paths))
(root / "provenance.json").write_text(
    json.dumps({"repository": repo, "revision": rev, "files": files}, indent=2)
)
