"""Prepare pinned FI / ArtPhoto photographs with exact duplicate safeguards.

Reads image members without extracting archives. Manifest hashes identify the
custom split; it is not the original-paper benchmark split. All graph labels
remain the same eight Mikels labels. No training or GPU work occurs here.
"""
from __future__ import annotations
import argparse,csv,hashlib,io,json,os,tarfile,time,urllib.request,urllib.parse,zipfile
from collections import Counter,defaultdict
from pathlib import Path,PurePosixPath
import numpy as np
from PIL import Image,ImageOps

REVISION='476a96fdfc8330626bf60ce686008bed5605e5c0'
REPOSITORY='wudq/Small-Scale_VSA_Datasets'
CLASSES=['amusement','anger','awe','contentment','disgust','excitement','fear','sadness']
SOURCES={
 'fi':('FI_dataset.tar','4c296245bdb31e6727b6aea806bada120a82b5651f966881956f2a62c45958da',2629120512),
 'artphoto':('testImages_artphoto.zip','70e261192a414ad50a02d359ec8712a3b0c5336d149aa671364ea0a74f0dae3c',20794350),
}
ALIASES={'amusement':'amusement','anger':'anger','angry':'anger','awe':'awe','contentment':'contentment','content':'contentment','disgust':'disgust','excitement':'excitement','exciting':'excitement','fear':'fear','sad':'sadness','sadness':'sadness'}
FIELDS=['path','label','source_id','image_sha256','pixel_sha256','resized_pixel_sha256','source_archive','source_member']

def sha256(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()

def download(dest,name,digest,size):
 dest.mkdir(parents=True,exist_ok=True);p=dest/name
 if p.exists():
  if p.stat().st_size != size or sha256(p)!=digest:raise ValueError(f'Existing archive invalid: {p}')
  return p
 temp=p.with_suffix(p.suffix+'.part')
 url=f'https://huggingface.co/datasets/{REPOSITORY}/resolve/{REVISION}/{urllib.parse.quote(name)}?download=true'
 print('Downloading',url,flush=True)
 with urllib.request.urlopen(url,timeout=90) as r,temp.open('wb') as f:
  done=0;next_log=32*1024**2
  while b:=r.read(4*1024**2):
   f.write(b);done+=len(b)
   if done>=next_log:print('Downloaded MB',round(done/1024**2),flush=True);next_log+=32*1024**2
 if temp.stat().st_size!=size or sha256(temp)!=digest:raise ValueError(f'Download hash/size mismatch: {temp}')
 os.replace(temp,p);return p

def members(path):
 if path.suffix=='.zip':
  with zipfile.ZipFile(path) as z:
   for info in sorted(z.infolist(),key=lambda i:i.filename):
    if not info.is_dir() and PurePosixPath(info.filename).suffix.lower() in {'.jpg','.jpeg','.png','.bmp'}:yield info.filename,z.read(info)
 else:
  with tarfile.open(path,'r:') as z:
   for info in z:
    if info.isfile() and PurePosixPath(info.name).suffix.lower() in {'.jpg','.jpeg','.png','.bmp'}:
     f=z.extractfile(info)
     if f:yield info.name,f.read()

def write_csv(path,rows):
 with path.open('w',newline='',encoding='utf8') as f:
  w=csv.DictWriter(f,fieldnames=FIELDS);w.writeheader();w.writerows(rows)

def exclusion_hashes(roots):
 encoded=set();pixels=set();resized=set();inventory=[]
 for root in roots:
  root=Path(root)
  for split in ('train','val','test','full'):
   p=root/f'{split}.csv'
   if not p.exists():continue
   rows=list(csv.DictReader(p.open(encoding='utf-8-sig')))
   encoded.update(r['image_sha256'] for r in rows if r.get('image_sha256'))
   pixels.update(r['pixel_sha256'] for r in rows if r.get('pixel_sha256'))
   resized.update(r['resized_pixel_sha256'] for r in rows if r.get('resized_pixel_sha256'))
   inventory.append({'path':str(p),'sha256':sha256(p),'n':len(rows)})
 return encoded,pixels,resized,inventory

def prepare(args,name):
 archive_name,digest,size=SOURCES[name];archive=download(args.downloads,archive_name,digest,size)
 root=args.out/name
 if (root/'manifest.json').exists():raise FileExistsError(f'Use a fresh dataset output to avoid silent overwrite: {root}')
 root.mkdir(parents=True,exist_ok=True);(root/'images').mkdir(exist_ok=True)
 other_roots=[args.out/n for n in SOURCES if n!=name and (args.out/n/'manifest.json').exists()]
 excluded_encoded,excluded_pixels,excluded_resized,exclusions=exclusion_hashes(args.exclude_roots+other_roots)
 seen_encoded=set();seen_pixels=set();seen_resized=set();rejected=[];rows=[];raw_counts=Counter();resize_duplicates=0;pixel_labels=defaultdict(set);member_pixels={}
 for index,(member,b) in enumerate(members(archive)):
  p=PurePosixPath(member)
  if p.is_absolute() or '..' in p.parts:raise ValueError(f'Unsafe member: {member}')
  # FI stores class directories; ArtPhoto stores class_name.jpg at one level.
  token=p.stem.split('_')[0].lower();parent=p.parent.name.lower()
  label=ALIASES.get(parent,ALIASES.get(token))
  if label is None:raise ValueError(f'Unknown label: {member}')
  raw_counts[label]+=1;encoded=hashlib.sha256(b).hexdigest()
  try:
   with Image.open(io.BytesIO(b)) as source:im=ImageOps.exif_transpose(source).convert('RGB')
   pixel=hashlib.sha256(str(im.size).encode()+im.tobytes()).hexdigest()
  except (OSError,ValueError,Image.DecompressionBombError) as e:
   rejected.append({'member':member,'reason':'invalid_image','detail':str(e)});continue
  resized_im=im.resize((192,192),Image.Resampling.BILINEAR)
  resized=hashlib.sha256(str(resized_im.size).encode()+resized_im.tobytes()).hexdigest()
  pixel_labels[pixel].add(label);member_pixels[member]=pixel
  if resized in seen_resized:resize_duplicates+=1
  reason=None
  if encoded in seen_encoded or pixel in seen_pixels:reason='within_dataset_exact_duplicate'
  elif encoded in excluded_encoded or pixel in excluded_pixels or resized in excluded_resized:reason='external_exact_overlap'
  elif resized in seen_resized:reason='within_dataset_identical_resized_input'
  # Mark all valid images seen, so even differently labeled duplicate images
  # cannot enter separate partitions.
  seen_encoded.add(encoded);seen_pixels.add(pixel);seen_resized.add(resized)
  if reason:rejected.append({'member':member,'reason':reason});continue
  # PNG preserves the resized pixels; no secondary JPEG compression confound.
  filename=f'images/{pixel}.png'
  resized_im.save(root/filename)
  rows.append(dict(path=filename,label=label,source_id=p.stem,image_sha256=encoded,pixel_sha256=pixel,resized_pixel_sha256=resized,source_archive=archive_name,source_member=member))
  if index%1000==0:print(name,'scanned',index,'retained',len(rows),flush=True)
 # Resolve ambiguous duplicate groups before assigning any split. Retaining
 # the first archive member would attach an arbitrary label to identical pixels.
 ambiguous={pixel for pixel,labels in pixel_labels.items() if len(labels)>1}
 ambiguity_audit=[{'pixel_sha256':pixel,'labels':sorted(pixel_labels[pixel]),
   'members':[member for member,h in member_pixels.items() if h==pixel]}
   for pixel in sorted(ambiguous)]
 for rejection in rejected:
  if member_pixels.get(rejection['member']) in ambiguous:
   rejection['reason']='ambiguous_crosslabel_pixel_group'
 for row in rows:
  if row['pixel_sha256'] in ambiguous:
   rejected.append({'member':row['source_member'],'reason':'ambiguous_crosslabel_pixel_group'})
 rows=[row for row in rows if row['pixel_sha256'] not in ambiguous]
 (root/'ambiguity_audit.json').write_text(json.dumps(ambiguity_audit,indent=2),encoding='utf8')
 # Splits are selected once with an explicit data-split seed. Training seeds
 # change only initialization/order/augmentation, not these assignments.
 rng=np.random.default_rng(args.split_seed);splits={s:[] for s in ('train','val','test')}
 for c in CLASSES:
  rr=sorted([r for r in rows if r['label']==c],key=lambda r:r['pixel_sha256'])
  rr=[rr[int(i)] for i in rng.permutation(len(rr))]
  ntrain=int(len(rr)*.6);nval=int(len(rr)*.2)
  if nval<2 or len(rr)-ntrain-nval<2:raise ValueError(f'Insufficient class {c}: {len(rr)}')
  partitions={'train':rr[:ntrain],'val':rr[ntrain:ntrain+nval],'test':rr[ntrain+nval:]}
  for s,subset in partitions.items():
   cap=getattr(args,s+'_per_class')
   if cap is None:cap={'train':600,'val':100,'test':200}[s] if name=='fi' else 0
   splits[s].extend(subset if cap==0 else subset[:cap])
 for split,rr in splits.items():write_csv(root/f'{split}.csv',rr)
 write_csv(root/'full.csv',rows)
 (root/'classes.txt').write_text('\n'.join(CLASSES)+'\n',encoding='utf8')
 (root/'excluded.json').write_text(json.dumps(rejected,indent=2),encoding='utf8')
 manifest={'dataset':name,'mirror':REPOSITORY,'revision':REVISION,'archive':archive_name,'archive_sha256':digest,'archive_size':size,'source_url':f'https://huggingface.co/datasets/{REPOSITORY}/resolve/{REVISION}/{archive_name}',
  'official_source':'https://ojs.aaai.org/index.php/AAAI/article/view/9987' if name=='fi' else 'https://imageemotion.org/',
  'license_note':'Mirror supplies no dataset license card; original photographs may be copyright. Academic research use only; do not redistribute raw photos. ArtPhoto original site explicitly limits scientific use.',
  'split_seed':args.split_seed,'split_rule':'60/20/20 stratified by class after exact deduplication; optional per-class caps; fixed custom split, not official paper split','pixel_hash_rule':'sha256(str(EXIF-transposed RGB size).encode()+RGBbytes) before resize','image_transform':'PIL bilinear resize192x192; PNG lossless storage','raw_counts':dict(raw_counts),'retained_counts':dict(Counter(r['label'] for r in rows)),'rejected_counts':dict(Counter(r['reason'] for r in rejected)),'identical_resized_input_diagnostic':resize_duplicates,'ambiguous_crosslabel_pixel_groups':len(ambiguous),'ambiguity_policy':'All members of every decoded-pixel group with conflicting class labels excluded before split assignment',
  'excluded_sources':exclusions,'exclusion_scope':'Only encoded hashes provided by full EmoSet manifests and original pixel hashes provided by pilot manifests are checked. Re-encoded/near duplicates and other unprovided source images remain possible.',
  'splits':{s:{'n':len(rr),'per_class':dict(Counter(r['label'] for r in rr)),'csv_sha256':sha256(root/f'{s}.csv')} for s,rr in splits.items()}}
 (root/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8');print(json.dumps(manifest,indent=2),flush=True)

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--datasets',nargs='+',choices=SOURCES,default=['fi','artphoto']);p.add_argument('--downloads',type=Path,default=Path('data/external_source'));p.add_argument('--out',type=Path,default=Path('data/seed_audit'));p.add_argument('--exclude-roots',nargs='*',default=['data/emoset_graph','data/kaggle_full_8epoch/data/emoset_full']);p.add_argument('--split-seed',type=int,default=20261117)
 for split in ('train','val','test'):p.add_argument('--'+split+'-per-class',type=int,default=None)
 args=p.parse_args()
 for s in ('train','val','test'):
  if getattr(args,s+'_per_class') is not None and getattr(args,s+'_per_class')<0:p.error('Class caps must be nonnegative')
 for name in args.datasets:prepare(args,name)
if __name__=='__main__':main()

