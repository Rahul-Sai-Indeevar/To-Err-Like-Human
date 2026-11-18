"""Apply a pre-outcome ambiguity audit to a prepared dataset without image decoding.

Preserves all original CSV/JSON inputs under pre_ambiguity_cleanup before
atomically replacing manifests. No image is deleted or rewritten. This utility
is for preparation artifacts made before the all-members ambiguity policy.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
from collections import Counter
import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_text(path, value):
    temporary = path.with_name(path.name + '.cleanup.part')
    temporary.write_text(value, encoding='utf8', newline='')
    os.replace(temporary, path)


def csv_text(rows, fields):
    output = io.StringIO(newline='')
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def repair(root, audit_path, caps=(600,100,200)):
    root = Path(root).resolve()
    audit_path = Path(audit_path).resolve()
    manifest_path = root / 'manifest.json'
    original_manifest = json.loads(manifest_path.read_text(encoding='utf8'))
    audit = json.loads(audit_path.read_text(encoding='utf8'))
    if audit['dataset'] != original_manifest['dataset']:
        raise ValueError('Audit names a different dataset')
    if digest(root / 'full.csv') != audit['source_full_csv_sha256']:
        raise ValueError('Audit does not identify this exact full.csv')
    if 'ambiguity_cleanup' in original_manifest:
        raise ValueError('Ambiguity cleanup has already been applied')
    with (root / 'full.csv').open(encoding='utf8', newline='') as stream:
        reader=csv.DictReader(stream);fields=reader.fieldnames;rows=list(reader)
    rejected = json.loads((root / 'excluded.json').read_text(encoding='utf8'))
    ambiguous_hashes={group['pixel_sha256'] for group in audit['ambiguous_groups']}
    ambiguous_members={member for group in audit['ambiguous_groups'] for member in group['members']}
    removed=[row for row in rows if row['pixel_sha256'] in ambiguous_hashes]
    expected_members={row['source_member'] for row in removed}
    if not expected_members <= ambiguous_members:
        raise ValueError('Audit lacks a retained member of an ambiguous group')
    for group in audit['ambiguous_groups']:
        if len(group['labels']) < 2 or group['n_members'] != len(group['members']):
            raise ValueError('Malformed ambiguity group')
    filtered=[row for row in rows if row['pixel_sha256'] not in ambiguous_hashes]
    revised_rejected=[]
    for row in rejected:
        row=dict(row)
        if row['member'] in ambiguous_members:
            row['reason']='ambiguous_crosslabel_pixel_group'
        revised_rejected.append(row)
    revised_rejected.extend({'member':row['source_member'],'reason':'ambiguous_crosslabel_pixel_group'} for row in removed)
    actual_ambiguous={row['member'] for row in revised_rejected if row['reason']=='ambiguous_crosslabel_pixel_group'}
    if actual_ambiguous != ambiguous_members:
        raise ValueError('Audit members do not reconcile with original full/excluded rows')
    classes=(root/'classes.txt').read_text(encoding='utf8').splitlines()
    rng=np.random.default_rng(original_manifest['split_seed'])
    splits={s:[] for s in ('train','val','test')}
    for label in classes:
        group=sorted([row for row in filtered if row['label']==label],key=lambda r:r['pixel_sha256'])
        group=[group[int(i)] for i in rng.permutation(len(group))]
        ntrain=int(len(group)*.6);nval=int(len(group)*.2)
        if nval<2 or len(group)-ntrain-nval<2:
            raise ValueError(f'Insufficient retained class: {label}')
        parts=[group[:ntrain],group[ntrain:ntrain+nval],group[ntrain+nval:]]
        for split,part,cap in zip(splits,parts,caps):
            splits[split].extend(part if cap==0 else part[:cap])
    # Validate everything before preserving originals or replacing a manifest.
    all_hashes=[row['pixel_sha256'] for values in splits.values() for row in values]
    if len(all_hashes) != len(set(all_hashes)):
        raise ValueError('Replacement splits overlap')
    backup=root/'pre_ambiguity_cleanup'
    backup.mkdir(exist_ok=False)
    names=['full.csv','train.csv','val.csv','test.csv','manifest.json','excluded.json']
    backup_inventory={}
    for name in names:
        source=root/name
        shutil.copy2(source,backup/name)
        backup_inventory[name]=digest(source)
    atomic_text(backup/'inventory.json',json.dumps(backup_inventory,indent=2))
    for split,values in splits.items():
        atomic_text(root/f'{split}.csv',csv_text(values,fields))
    atomic_text(root/'full.csv',csv_text(filtered,fields))
    atomic_text(root/'excluded.json',json.dumps(revised_rejected,indent=2))
    atomic_text(root/'ambiguity_audit.json',json.dumps(audit,indent=2))
    manifest=dict(original_manifest)
    manifest['retained_counts']=dict(Counter(row['label'] for row in filtered))
    manifest['rejected_counts']=dict(Counter(row['reason'] for row in revised_rejected))
    manifest['ambiguous_crosslabel_pixel_groups']=len(ambiguous_hashes)
    manifest['ambiguity_policy']='All members of every decoded-pixel group with conflicting class labels excluded before split assignment'
    manifest['ambiguity_cleanup']={
        'original_manifest_sha256':backup_inventory['manifest.json'],
        'original_full_csv_sha256':backup_inventory['full.csv'],
        'audit_sha256':digest(audit_path),
        'newly_removed_unique_rows':len(removed),
        'raw_ambiguous_members_excluded':len(ambiguous_members),
        'backup_directory':backup.name,
        'split_rule_unchanged':'60/20/20 stratified by class after deduplication, then per-class caps',
        'caps':dict(zip(splits,caps)),
        'note':'The existing first-label duplicate policy was corrected at data preparation, before any FI model outcomes were observed. Original artifacts remain preserved.'
    }
    manifest['splits']={s:{'n':len(rr),'per_class':dict(Counter(r['label'] for r in rr)),
        'csv_sha256':digest(root/f'{s}.csv')} for s,rr in splits.items()}
    # Manifest is last, so a complete manifest never advertises unwritten CSVs.
    atomic_text(manifest_path,json.dumps(manifest,indent=2))
    print(json.dumps({'removed_unique_rows':len(removed),'total_remaining':len(filtered),
        'raw_ambiguous_rows':len(ambiguous_members),'splits':manifest['splits']},indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--audit',type=Path,required=True)
    parser.add_argument('--caps',type=int,nargs=3,default=[600,100,200])
    args=parser.parse_args()
    if min(args.caps)<0:parser.error('Caps must be nonnegative')
    repair(args.root,args.audit,args.caps)


if __name__=='__main__':main()
