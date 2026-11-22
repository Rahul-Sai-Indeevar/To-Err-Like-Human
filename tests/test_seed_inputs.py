"""CPU audit must detect content errors despite valid CSV/cache-key metadata."""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest


def _load_verifier():
    path=Path(__file__).resolve().parents[1]/'scripts'/'verify_seed_inputs.py'
    if not path.exists():path=Path(__file__).with_name('verify_seed_inputs.py')
    spec=importlib.util.spec_from_file_location('input_verifier',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


VERIFY=_load_verifier()


def _fixture(tmp_path, *, wrong_declared_hash=False):
    root=tmp_path/'data'/'synthetic';root.mkdir(parents=True)
    classes=['amusement','sadness'];(root/'classes.txt').write_text('\n'.join(classes)+'\n')
    (root/'manifest.json').write_text('{}')
    rows=[];arrays=[]
    for index,label in enumerate(classes):
        # Known constant signals make an exact pixel comparison straightforward.
        pixels=np.full((192,192,3),35+100*index,dtype=np.uint8)
        image=Image.fromarray(pixels);image.save(root/f'image{index}.png')
        resized_hash=hashlib.sha256(str((192,192)).encode()+pixels.tobytes()).hexdigest()
        if wrong_declared_hash and index==0:resized_hash='0'*64
        rows.append({'path':f'image{index}.png','label':label,'resized_pixel_sha256':resized_hash})
        arrays.append(pixels.transpose(2,0,1))
    with (root/'train.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
    csv_bytes=(root/'train.csv').read_bytes()
    key=hashlib.sha256(csv_bytes+VERIFY.CACHE_TRANSFORM_TAG).hexdigest()
    cache=root/f'train_seed_evidence_{key[:16]}.npy'
    np.save(cache,np.stack(arrays))
    protocol={'args':{'data_root':'data/synthetic'},'classes':classes,
        'datasets':{'train':{'n':2,'csv_sha256':hashlib.sha256(csv_bytes).hexdigest(),
            'cache_key':key,'class_counts':[1,1]}},
        'data_manifest_sha256':hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest()}
    protocol['protocol_hash']=hashlib.sha256(VERIFY.canonical(protocol)).hexdigest()
    path=tmp_path/'protocol.json';path.write_text(json.dumps(protocol))
    return path,cache


def test_full_exact_image_cache_alignment_and_cache_file_identity(tmp_path):
    protocol,cache=_fixture(tmp_path)
    result=VERIFY.verify_protocol(protocol,tmp_path)
    split=result['splits']['train']
    assert split['rows_verified']==split['exact_source_vs_cache_matches']==2
    assert split['declared_resized_pixel_hashes_verified']==2
    assert split['cache_npy_sha256']==hashlib.sha256(cache.read_bytes()).hexdigest()
    assert split['shape']==[2,3,192,192]


def test_tampered_cache_detected_with_unchanged_valid_csv_key(tmp_path):
    protocol,cache=_fixture(tmp_path)
    data=np.load(cache,mmap_mode='r+');data[0,0,0,0]^=1;data.flush();del data
    with pytest.raises(ValueError,match='train row 0: cache/image mismatch'):
        VERIFY.verify_protocol(protocol,tmp_path)


def test_wrong_declared_resized_hash_detected_even_when_cache_matches_image(tmp_path):
    protocol,_=_fixture(tmp_path,wrong_declared_hash=True)
    with pytest.raises(ValueError,match='declared resized pixel hash mismatch'):
        VERIFY.verify_protocol(protocol,tmp_path)


def test_recorded_cache_file_checksum_is_verified_independently(tmp_path):
    protocol,_=_fixture(tmp_path)
    contents=json.loads(protocol.read_text())
    contents['datasets']['train']['cache_sha256']='0'*64
    contents.pop('protocol_hash')
    contents['protocol_hash']=hashlib.sha256(VERIFY.canonical(contents)).hexdigest()
    protocol.write_text(json.dumps(contents))
    with pytest.raises(ValueError,match='cache file SHA256 differs from recorded training protocol'):
        VERIFY.verify_protocol(protocol,tmp_path)


def test_full_audit_rejects_benchmark_or_incomplete_protocol(tmp_path):
    protocol,_=_fixture(tmp_path)
    with pytest.raises(ValueError,match='Expected full training protocol'):
        VERIFY.verify_protocol(protocol,tmp_path,require_full_splits=True)


def test_full_audit_rejects_explicit_skip_test_even_if_three_splits_advertised(tmp_path):
    protocol,_=_fixture(tmp_path)
    contents=json.loads(protocol.read_text())
    contents['args']['skip_test']=True
    contents['datasets']['val']=dict(contents['datasets']['train'])
    contents['datasets']['test']=dict(contents['datasets']['train'])
    contents.pop('protocol_hash')
    contents['protocol_hash']=hashlib.sha256(VERIFY.canonical(contents)).hexdigest()
    protocol.write_text(json.dumps(contents))
    with pytest.raises(ValueError,match='Expected full training protocol'):
        VERIFY.verify_protocol(protocol,tmp_path,require_full_splits=True)


def test_require_full_splits_cli_rejects_incomplete_protocol_without_success_json(tmp_path):
    import subprocess
    import sys
    protocol,_=_fixture(tmp_path)
    output=tmp_path/'audit.json'
    result=subprocess.run([sys.executable,str(Path(VERIFY.__file__).resolve()),
        '--repo-root',str(tmp_path),'--protocol',str(protocol),
        '--require-full-splits','--out',str(output)],capture_output=True,text=True)
    assert result.returncode!=0
    assert 'Expected full training protocol' in result.stderr
    assert not output.exists()
