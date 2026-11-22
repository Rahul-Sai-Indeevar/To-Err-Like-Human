"""CPU-only checks for data leakage, deterministic splits and safe preparation."""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import io
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
import zipfile

import numpy as np
from PIL import Image
import pytest


def _load_preparer():
    # The fallback lets the same tests run from the review staging directory.
    root = Path(__file__).resolve().parents[1]
    path = root / 'scripts' / 'prepare_seed_datasets.py'
    if not path.exists():
        path = Path(__file__).with_name('prepare_external.py')
    spec = importlib.util.spec_from_file_location('seed_dataset_preparer', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PREPARE = _load_preparer()


def _image_bytes(class_index, index, *, compress_level=6):
    # Random pixels, rather than constant images, also exercise resize hashing.
    rng = np.random.default_rng(10000 * class_index + index)
    pixels = rng.integers(0, 256, (17, 19, 3), dtype=np.uint8)
    image = Image.fromarray(pixels)
    buffer = io.BytesIO()
    image.save(buffer, format='PNG', compress_level=compress_level)
    pixel_hash = hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()
    return buffer.getvalue(), pixel_hash


def _synthetic_archive(path, *, conflicting_duplicate=False):
    hashes = {}
    with zipfile.ZipFile(path, 'w') as archive:
        for class_index, label in enumerate(PREPARE.CLASSES):
            for index in range(20):
                encoded, pixel_hash = _image_bytes(class_index, index)
                name = f'images/{label}/{label}_{index:03d}.png'
                archive.writestr(name, encoded)
                hashes[(label, index)] = pixel_hash
        if conflicting_duplicate:
            # Same decoded pixels, a different encoded PNG and a contradictory
            # label. Deduplicating bytes alone would leave this leakage intact.
            encoded, _ = _image_bytes(0, 0, compress_level=0)
            original, _ = _image_bytes(0, 0)
            assert encoded != original
            archive.writestr('images/anger/anger_conflict.png', encoded)
    return hashes


def _args(tmp_path, exclude_roots=()):
    return SimpleNamespace(
        out=tmp_path / 'prepared', downloads=tmp_path,
        exclude_roots=list(exclude_roots), split_seed=20261117,
        train_per_class=8, val_per_class=3, test_per_class=3,
    )


def _read_rows(path):
    with path.open(newline='', encoding='utf8') as stream:
        return list(csv.DictReader(stream))


def _write_external_csv(path, pixel_hash):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf8') as stream:
        writer = csv.DictWriter(stream, fieldnames=['pixel_sha256'])
        writer.writeheader()
        writer.writerow({'pixel_sha256': pixel_hash})


def test_exact_dedup_precedes_disjoint_deterministic_class_capped_splits(tmp_path, monkeypatch):
    archive = tmp_path / 'synthetic.zip'
    hashes = _synthetic_archive(archive, conflicting_duplicate=True)
    external = tmp_path / 'external'
    excluded_hash = hashes[('amusement', 2)]
    _write_external_csv(external / 'train.csv', excluded_hash)
    monkeypatch.setattr(PREPARE, 'download', lambda *args: archive)
    args = _args(tmp_path, [external])
    PREPARE.prepare(args, 'fi')
    root = args.out / 'fi'
    all_rows = _read_rows(root / 'full.csv')
    # 160 unique generated images, minus one external overlap and BOTH
    # variants of the ambiguous cross-label group, including its first member.
    assert len(all_rows) == 158
    assert len({row['pixel_sha256'] for row in all_rows}) == len(all_rows)
    assert excluded_hash not in {row['pixel_sha256'] for row in all_rows}
    assert not any(row['source_id'] == 'anger_conflict' for row in all_rows)
    assert hashes[('amusement', 0)] not in {row['pixel_sha256'] for row in all_rows}
    rejected = json.loads((root / 'excluded.json').read_text(encoding='utf8'))
    assert Counter(row['reason'] for row in rejected) == {
        'ambiguous_crosslabel_pixel_group': 2, 'external_exact_overlap': 1,
    }
    split_hashes = {}
    for split, cap in [('train', 8), ('val', 3), ('test', 3)]:
        rows = _read_rows(root / f'{split}.csv')
        assert Counter(row['label'] for row in rows) == {c: cap for c in PREPARE.CLASSES}
        split_hashes[split] = {row['pixel_sha256'] for row in rows}
        for row in rows:
            assert (root / row['path']).is_file()
            with Image.open(root / row['path']) as image:
                assert image.size == (192, 192)
    assert not split_hashes['train'] & split_hashes['val']
    assert not split_hashes['train'] & split_hashes['test']
    assert not split_hashes['val'] & split_hashes['test']
    # A second output destination must give exactly the same assignments;
    # changing the training seed must never secretly create new data splits.
    args.out = tmp_path / 'repeat'
    PREPARE.prepare(args, 'fi')
    for split in ['train', 'val', 'test']:
        assert (root / f'{split}.csv').read_bytes() == (args.out / 'fi' / f'{split}.csv').read_bytes()
    with pytest.raises(FileExistsError, match='fresh dataset output'):
        PREPARE.prepare(args, 'fi')


def test_previous_dataset_reserve_is_excluded_from_new_dataset(tmp_path, monkeypatch):
    archive = tmp_path / 'synthetic.zip'
    hashes = _synthetic_archive(archive)
    args = _args(tmp_path)
    # This image belongs to a previously prepared FI reserve, not a train/val/
    # test row. It must still be rejected from ArtPhoto cross-dataset analysis.
    prior = args.out / 'fi'
    _write_external_csv(prior / 'full.csv', hashes[('sadness', 19)])
    (prior / 'manifest.json').write_text('{}', encoding='utf8')
    monkeypatch.setattr(PREPARE, 'download', lambda *args: archive)
    PREPARE.prepare(args, 'artphoto')
    root = args.out / 'artphoto'
    all_rows = _read_rows(root / 'full.csv')
    assert len(all_rows) == 159
    assert hashes[('sadness', 19)] not in {row['pixel_sha256'] for row in all_rows}
    excluded = json.loads((root / 'excluded.json').read_text(encoding='utf8'))
    assert excluded == [{'member': 'images/sadness/sadness_019.png', 'reason': 'external_exact_overlap'}]


def test_unsafe_image_member_rejected_without_extracting(tmp_path, monkeypatch):
    archive = tmp_path / 'unsafe.zip'
    with zipfile.ZipFile(archive, 'w') as stream:
        stream.writestr('../amusement_000.png', _image_bytes(0, 0)[0])
    monkeypatch.setattr(PREPARE, 'download', lambda *args: archive)
    args = _args(tmp_path)
    with pytest.raises(ValueError, match='Unsafe member'):
        PREPARE.prepare(args, 'fi')
    assert not (tmp_path / 'amusement_000.png').exists()
    assert not list((args.out / 'fi' / 'images').glob('*'))
