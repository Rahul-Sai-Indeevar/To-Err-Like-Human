"""Independently verify source-image, CSV and uint8 cache alignment (CPU only).

Uses NumPy/Pillow, never imports the training runner or initializes CUDA. Checks
EVERY source image against the corresponding row of the exact protocol-named
training cache. JSON contains cache-file hashes and ordered source-image hash
aggregates, so the audit identifies more than CSV metadata alone.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path
import time
from collections import Counter

import numpy as np
from PIL import Image, ImageOps, __version__ as PIL_VERSION

CACHE_TRANSFORM_TAG = b'PIL_EXIF_RGB_torchvisionResize192_antialias_v1'


def sha256_file(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024**2),b''):
            value.update(block)
    return value.hexdigest()


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':')).encode()


def pixel_sha(image):
    return hashlib.sha256(str(image.size).encode()+image.tobytes()).hexdigest()


def verify_protocol(protocol_path, repo_root, require_full_splits=False):
    protocol_path=Path(protocol_path).resolve()
    repo_root=Path(repo_root).resolve()
    started=time.perf_counter()
    protocol=json.loads(protocol_path.read_text(encoding='utf8'))
    if require_full_splits and (protocol.get('args',{}).get('skip_test',False) or set(protocol['datasets'])!={'train','val','test'}):
        raise ValueError('Expected full training protocol with train/val/test; benchmark skip-test protocol is not accepted')
    advertised=protocol.get('protocol_hash')
    if advertised:
        unhashed={k:v for k,v in protocol.items() if k!='protocol_hash'}
        if hashlib.sha256(canonical(unhashed)).hexdigest()!=advertised:
            raise ValueError('Protocol hash does not match contents')
    data_root=(repo_root/protocol['args']['data_root']).resolve()
    classes=protocol['classes']
    actual_classes=(data_root/'classes.txt').read_text(encoding='utf8').splitlines()
    if classes!=actual_classes:
        raise ValueError('Class order differs between protocol and classes.txt')
    manifest=data_root/'manifest.json'
    if protocol.get('data_manifest_sha256') is not None:
        if sha256_file(manifest)!=protocol['data_manifest_sha256']:
            raise ValueError('Dataset manifest differs from protocol')
    result={'protocol_path':str(protocol_path),'protocol_file_sha256':sha256_file(protocol_path),
        'protocol_hash':advertised,'data_root':str(data_root),'classes':classes,'splits':{}}
    for split,metadata in protocol['datasets'].items():
        csv_path=data_root/f'{split}.csv'
        encoded_csv=csv_path.read_bytes()
        csv_digest=hashlib.sha256(encoded_csv).hexdigest()
        if csv_digest!=metadata['csv_sha256']:
            raise ValueError(f'{split}: CSV differs from protocol')
        cache_key=hashlib.sha256(encoded_csv+CACHE_TRANSFORM_TAG).hexdigest()
        if cache_key!=metadata['cache_key']:
            raise ValueError(f'{split}: cache key differs from independently computed transform key')
        with csv_path.open(encoding='utf-8-sig',newline='') as stream:
            rows=list(csv.DictReader(stream))
        labels=[classes.index(row['label'].lower()) for row in rows]
        counts=np.bincount(labels,minlength=len(classes)).tolist()
        if len(rows)!=metadata['n'] or counts!=metadata['class_counts']:
            raise ValueError(f'{split}: count/labels differ from protocol')
        cache_path=data_root/f'{split}_seed_evidence_{cache_key[:16]}.npy'
        before=cache_path.stat()
        cache=np.load(cache_path,mmap_mode='r',allow_pickle=False)
        if cache.dtype!=np.uint8 or cache.shape!=(len(rows),3,192,192):
            raise ValueError(f'{split}: cache shape/dtype mismatch')
        image_sequence=hashlib.sha256();array_sequence=hashlib.sha256()
        resized_hashes_checked=0;encoded_hashes_checked=0
        for index,row in enumerate(rows):
            path=(data_root/row['path']).resolve()
            if not path.is_relative_to(data_root):
                raise ValueError(f'{split} row {index}: source path escapes dataset')
            actual_encoded=sha256_file(path)
            with Image.open(path) as source:
                rgb=ImageOps.exif_transpose(source).convert('RGB')
                resized=rgb.resize((192,192),Image.Resampling.BILINEAR)
            # FI/ArtPhoto retain exact resized-pixel provenance. EmoSet's older
            # pixel_sha256 refers to pre-thumbnail originals, so is not falsely
            # compared to its subsequently saved/resized JPEG photograph.
            if row.get('resized_pixel_sha256'):
                if pixel_sha(resized)!=row['resized_pixel_sha256']:
                    raise ValueError(f'{split} row {index}: declared resized pixel hash mismatch')
                resized_hashes_checked+=1
            if row.get('prepared_image_sha256'):
                if actual_encoded!=row['prepared_image_sha256']:
                    raise ValueError(f'{split} row {index}: prepared image byte hash mismatch')
                encoded_hashes_checked+=1
            expected=np.asarray(resized,dtype=np.uint8).transpose(2,0,1)
            if not np.array_equal(expected,cache[index]):
                different=int(np.count_nonzero(expected!=cache[index]))
                raise ValueError(f'{split} row {index}: cache/image mismatch ({different} channel-pixels)')
            record={'index':index,'path':row['path'],'label':row['label'],
                'encoded_image_sha256':actual_encoded,'resized_pixel_sha256':pixel_sha(resized)}
            image_sequence.update(canonical(record));image_sequence.update(b'\n')
            array_sequence.update(cache[index].tobytes())
            if index%2000==0:
                print(data_root.name,split,'verified',index,'/',len(rows),flush=True)
        cache_file_digest=sha256_file(cache_path)
        recorded_cache_digest=metadata.get('cache_sha256')
        if recorded_cache_digest is not None and recorded_cache_digest!=cache_file_digest:
            raise ValueError(f'{split}: cache file SHA256 differs from recorded training protocol')
        after=cache_path.stat()
        if before.st_size!=after.st_size or before.st_mtime_ns!=after.st_mtime_ns:
            raise ValueError(f'{split}: cache changed while audit was running')
        result['splits'][split]={'rows_verified':len(rows),'class_counts':counts,
            'csv_sha256':csv_digest,'cache_key':cache_key,'cache_path':str(cache_path),
            'cache_npy_sha256':cache_file_digest,'cache_npy_bytes':after.st_size,
            'recorded_cache_sha256_verified':recorded_cache_digest is not None,
            'ordered_source_image_records_sha256':image_sequence.hexdigest(),
            'ordered_chw_uint8_rows_sha256':array_sequence.hexdigest(),
            'exact_source_vs_cache_matches':len(rows),
            'declared_resized_pixel_hashes_verified':resized_hashes_checked,
            'declared_prepared_encoded_hashes_verified':encoded_hashes_checked,
            'shape':list(cache.shape),'dtype':str(cache.dtype)}
    result['seconds']=time.perf_counter()-started
    result['status']='all checked rows match exactly'
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root',type=Path,default=Path.cwd())
    parser.add_argument('--datasets',nargs='+',default=['emoset','fi','artphoto'])
    parser.add_argument('--protocol',type=Path,nargs='+',help='Explicit protocol paths, relative to repo root unless absolute; overrides --datasets')
    parser.add_argument('--require-full-splits',action='store_true',help='Reject benchmark/skip-test protocols')
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    reports={}
    if args.protocol:
        paths=[p if p.is_absolute() else args.repo_root/p for p in args.protocol]
        named=[(p.parent.name,p) for p in paths]
    else:
        named=[(name,args.repo_root/'outputs/seed_evidence'/name/'protocol.json') for name in args.datasets]
    for name,path in named:
        if name in reports:raise ValueError('Duplicate protocol output name; audit these protocols separately')
        reports[name]=verify_protocol(path,args.repo_root,args.require_full_splits)
    result={'audit':'CPU full source-image/CSV/cache alignment','python':platform.python_version(),
        'numpy':np.__version__,'pillow':PIL_VERSION,
        'implementation':'Independent NumPy/Pillow; EXIF-transposed RGB, PIL bilinear192, CHWuint8 exact equality; no runner import or GPU',
        'datasets':reports,'total_rows_verified':sum(s['rows_verified'] for r in reports.values() for s in r['splits'].values())}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2),encoding='utf8')
    print('VERIFIED',result['total_rows_verified'],'rows; JSON',args.out,flush=True)


if __name__=='__main__':main()
