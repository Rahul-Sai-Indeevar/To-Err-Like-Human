"""Dual-T4, three-seed graph coefficient sweep for To-Err-Like-Human.

The exact GitHub branch's graph loss, dataset cache, metrics, and emotion-wheel
are imported from the local repository; this script does not redefine them.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import shutil
import sys
import time
import traceback
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

REPO = Path(os.environ.get("GSP_REPO", "/kaggle/working/To-Err-Like-Human")).resolve()
EXP = Path(os.environ.get("GSP_OUTPUT", "/kaggle/working/graph_spectral_2gpu")).resolve()
sys.path.insert(0, str(REPO / "src"))
os.chdir(REPO)

from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision.models import ResNet18_Weights, resnet18
from ecc.common import predict_probs, set_seed
from ecc.emotion import build_cost_matrix, build_rank_matrix, polarity_vector
from ecc.graph import build_loss, graph_basis, heat_filter
from ecc.graph_experiment import Uint8Normalize
from ecc.metrics import confusion_matrix, evaluate_predictions, graph_error_metrics

ROOT = REPO / "data" / "emoset_full"
CACHE = Path(os.environ.get("GSP_CACHE", "/kaggle/temp/emoset_full_arrays"))
RUNS = EXP / "runs"
PLOTS = EXP / "plots"


# Alpha=0 is EXACTLY CE. Gamma and beta are held fixed to isolate alpha.
ALPHAS = (0.0, 0.1, 0.3, 0.5, 1.0)
def alpha_method(alpha):
    return "ce" if alpha == 0 else "graph_a" + format(alpha, ".1f").replace(".", "p")

METHODS = {
    alpha_method(a): (
        dict(kind="ce", alpha=0.0) if a == 0 else
        dict(kind="graph", alpha=float(a), gamma=1.0, beta=0.01, topology="complete")
    )
    for a in ALPHAS
}

def display_name(method):
    if method == "ce": return "CE (Î±=0)"
    return "Graph Î±=" + format(METHODS[method]["alpha"], ".1f")


def read_json(path, default=None):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else default


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2, default=float), encoding="utf-8")
    temp.replace(path)


def atomic_torch(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    torch.save(value, temp)
    temp.replace(path)


def load_classes():
    names = [c.strip() for c in (ROOT / "classes.txt").read_text().splitlines() if c.strip()]
    assert len(names) == 8, names
    return names


class FullEmotionDataset(Dataset):
    """Load all usable split rows; use memory-mapped RGB cache when available."""
    def __init__(self, split, classes, train=False):
        import csv
        self.split, self.train = split, train
        with (ROOT / f"{split}.csv").open(newline="") as f:
            rows=list(csv.DictReader(f))
        idx={c:i for i,c in enumerate(classes)}
        self.paths=[ROOT / row["path"] for row in rows]
        self.labels=np.array([idx[row["label"]] for row in rows],dtype=np.int64)
        self.array=None
        target=CACHE / f"{split}_192.npy"
        meta=read_json(CACHE / "metadata.json",{})
        if target.exists() and meta.get("manifest_digest")==hashlib.sha256((ROOT/"manifest.json").read_bytes()).hexdigest():
            loaded=np.load(target,mmap_mode="c")
            if loaded.shape==(len(rows),3,192,192):self.array=loaded

    def __len__(self):return len(self.labels)

    def __getitem__(self, i):
        if self.array is not None:
            x=torch.from_numpy(self.array[i])
        else:
            with Image.open(self.paths[i]) as im:
                x=torch.from_numpy(np.array(im.convert("RGB"),dtype=np.uint8).transpose(2,0,1).copy())
        if self.train:
            top,left=torch.randint(0,33,(2,)).tolist()
            x=x[:,top:top+160,left:left+160]
            if torch.rand(())<0.5:x=x.flip(-1)
        else:
            x=x[:,16:176,16:176]
        return x,int(self.labels[i])


def create_cache(classes):
    """Build a reusable ~13 GB uint8 scratch cache; never put it in saved working disk."""
    manifest_bytes=(ROOT/"manifest.json").read_bytes()
    digest=hashlib.sha256(manifest_bytes).hexdigest()
    previous=read_json(CACHE/"metadata.json",{})
    counts={s:sum(1 for _ in open(ROOT/f"{s}.csv",encoding="utf8"))-1 for s in ("train","val","test")}
    if previous.get("manifest_digest")==digest and all((CACHE/f"{s}_192.npy").exists() for s in counts):
        print("Scratch image cache already available.",flush=True);return
    bytes_needed=sum(counts.values())*3*192*192
    try:
        CACHE.mkdir(parents=True,exist_ok=True)
        free=shutil.disk_usage(CACHE).free
    except OSError:
        free=0
    print(f"Optional fast uint8 cache: need {bytes_needed/(1024**3):.2f} GiB; scratch free {free/(1024**3):.2f} GiB.",flush=True)
    if free < bytes_needed*1.18:
        print("Skipping memmap cache: using JPEG dataloading to avoid running out of disk.",flush=True)
        return
    for split,n in counts.items():
        path=CACHE/f"{split}_192.npy"
        if path.exists():path.unlink()
        dst=np.lib.format.open_memmap(path,mode="w+",dtype=np.uint8,shape=(n,3,192,192))
        with (ROOT/f"{split}.csv").open(newline="") as f:
            for i,row in enumerate(csv.DictReader(f)):
                with Image.open(ROOT/row["path"]) as im:
                    dst[i]=np.asarray(im.convert("RGB"),dtype=np.uint8).transpose(2,0,1)
                if i and i%10000==0:
                    print(f"Image cache {split}: {i:,}/{n:,}",flush=True)
        dst.flush();del dst
    atomic_json(CACHE/"metadata.json",{"manifest_digest":digest,"counts":counts})


def data_loaders(classes, batch, workers):
    out={}
    for name in ("train","val","test"):
        ds=FullEmotionDataset(name,classes,train=name=="train")
        actual_workers=workers if ds.array is not None else max(workers,2)
        out[name]=DataLoader(ds,batch_size=batch,shuffle=(name=="train"),num_workers=actual_workers,
                             pin_memory=True,persistent_workers=actual_workers>0,drop_last=False)
    return out


def model_init(seed):
    set_seed(seed)
    backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    backbone.fc = torch.nn.Linear(backbone.fc.in_features, 8)
    return torch.nn.Sequential(Uint8Normalize(), backbone).cuda()


def evaluate(model, loader, W, polarity, classes, use_amp):
    probs, targets = predict_probs(model, loader, torch.device("cuda"), amp=use_amp)
    prediction = probs.argmax(1)
    cm = confusion_matrix(targets, prediction, len(classes))
    basic = evaluate_predictions(targets, prediction, W, polarity)
    basic.update(graph_error_metrics(probs, targets, W))
    basic["n_errors"] = int((targets != prediction).sum())
    diag = np.diag(cm).astype(float)
    prec = diag / np.maximum(cm.sum(axis=0), 1)
    recall = diag / np.maximum(cm.sum(axis=1), 1)
    f1 = 2 * prec * recall / np.maximum(prec + recall, 1e-12)
    basic["macro_f1"] = float(f1.mean() * 100)
    basic["per_class"] = [
        dict(label=c, precision=float(prec[i]*100), recall=float(recall[i]*100),
             f1=float(f1[i]*100), support=int(cm[i].sum()))
        for i,c in enumerate(classes)
    ]
    confidence = probs.max(1)
    correct = (prediction == targets).astype(float)
    edges = np.linspace(0, 1, 11)
    calibration = []
    ece = 0.0
    for j in range(10):
        mask = (confidence >= edges[j]) & ((confidence <= edges[j+1]) if j == 9 else (confidence < edges[j+1]))
        count = int(mask.sum())
        acc = float(correct[mask].mean()) if count else None
        avg_conf = float(confidence[mask].mean()) if count else None
        if count:
            ece += count / len(confidence) * abs(acc - avg_conf)
        calibration.append(dict(left=float(edges[j]), right=float(edges[j+1]), count=count,
                                accuracy=acc, confidence=avg_conf))
    basic["ece"] = float(100 * ece)
    basic["calibration"] = calibration
    try:
        from sklearn.metrics import roc_auc_score
        basic["macro_ovr_auc"] = float(roc_auc_score(targets, probs, multi_class="ovr", average="macro"))
    except (ImportError, ValueError):
        basic["macro_ovr_auc"] = None
    return basic, probs, targets, cm


def training_job(method, seed, config, deadline):
    epochs = config["epochs"]
    batch = config["batch_size"]
    workers = config["num_workers"]
    d = RUNS / f"{method}_seed{seed}"
    d.mkdir(parents=True, exist_ok=True)
    final_path = d / "metrics.json"
    done = read_json(final_path)
    if done and done.get("completed_epochs", 0) >= epochs:
        print(f"[SKIP] {method} seed={seed}: already completed {epochs} epochs", flush=True)
        return
    classes = load_classes()
    W = build_cost_matrix(classes)
    rank = build_rank_matrix(W)
    polarity = polarity_vector(classes)
    set_seed(seed)
    load = data_loaders(classes, batch, workers)
    model = model_init(seed)
    criterion = build_loss(METHODS[method], W, rank).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    amp = bool(config["amp"])
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    current = 0
    best_ecc = -float("inf")
    best_epoch = 0
    elapsed = 0.0
    history_file = d / "history.json"
    history = read_json(history_file, [])
    saved = d / "last_resume.pt"
    if saved.exists():
        ck = torch.load(saved, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        if ck.get("scaler"):
            scaler.load_state_dict(ck["scaler"])
        current, best_ecc, best_epoch = ck["epoch"], ck["best_ecc"], ck["best_epoch"]
        elapsed = float(ck.get("elapsed", 0.0))
        print(f"[RESUME] {method} seed={seed} at epoch {current}/{epochs}", flush=True)
    meta = dict(method=method, seed=seed, loss=METHODS[method], config=config)
    atomic_json(d / "run_config.json", meta)
    run_start = time.time()
    for epoch in range(current, epochs):
        if time.time() >= deadline - config["reserve_seconds"]:
            print(f"[BUDGET] {method} seed={seed}, stopping at epoch {epoch}", flush=True)
            break
        model.train()
        epoch_start = time.perf_counter()
        weighted_total = 0.0
        seen = 0
        for x,y in load["train"]:
            x = x.cuda(non_blocking=True)
            y = y.cuda(non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=amp):
                logits = model(x)
            # Keep the graph eigenspace loss in float32 even when the ResNet uses AMP.
            with torch.autocast(device_type="cuda", enabled=False):
                loss = criterion(logits.float(), y)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss at {method} seed{seed} epoch{epoch+1}")
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            weighted_total += float(loss.item()) * len(y)
            seen += len(y)
        valp, valy = predict_probs(model, load["val"], torch.device("cuda"), amp=amp)
        valm = evaluate_predictions(valy, valp.argmax(1), W, polarity)
        valm.update(graph_error_metrics(valp, valy, W))
        duration = time.perf_counter() - epoch_start
        line = dict(epoch=epoch+1, train_loss=weighted_total/max(seen, 1),
                    val_acc=valm["acc"],val_ecc=valm["ecc"],val_emc=valm["emc"],
                    val_acc2=valm["acc2"],val_gsee=valm["gsee"],seconds=duration)
        history.append(line)
        atomic_json(history_file, history)
        if valm["ecc"] > best_ecc:
            best_ecc, best_epoch = valm["ecc"], epoch+1
            atomic_torch(d / "best.pt", model.state_dict())
        current = epoch+1
        atomic_torch(saved, dict(model=model.state_dict(), optimizer=opt.state_dict(),
                                 scaler=scaler.state_dict(), epoch=current,
                                 best_ecc=best_ecc, best_epoch=best_epoch,
                                 elapsed=elapsed + time.time()-run_start))
        print(f"[{method} seed{seed}] epoch {current}/{epochs} | train {line['train_loss']:.4f} "
              f"| val ACC {valm['acc']:.2f} ECC {valm['ecc']:.2f} | {duration:.1f}s",
              flush=True)
    if not (d / "best.pt").exists():
        raise RuntimeError(f"No best checkpoint saved: {d}")
    model.load_state_dict(torch.load(d / "best.pt", map_location="cpu", weights_only=True))
    model.cuda()
    valp, valy = predict_probs(model, load["val"], torch.device("cuda"), amp=amp)
    choices = [0.0, 0.1, 0.3, 1.0]
    val_scores = [evaluate_predictions(valy, heat_filter(valp,W,t).argmax(1), W, polarity)["ecc"]
                  for t in choices]
    chosen = choices[int(np.argmax(val_scores))]
    testm, p, y, cm = evaluate(model, load["test"], W, polarity, classes, amp)
    heat_p = heat_filter(p, W, chosen)
    heat_scores = evaluate_predictions(y, heat_p.argmax(1), W, polarity)
    np.savez_compressed(d / "predictions.npz", probs=p.astype(np.float32),
                        labels=y.astype(np.int16), heat_probs=heat_p.astype(np.float32))
    np.savetxt(d / "confusion_matrix.csv", cm, fmt="%d", delimiter=",")
    with (d / "per_class.csv").open("w", newline="") as f:
        cw = csv.DictWriter(f, fieldnames=["label","precision","recall","f1","support"])
        cw.writeheader();cw.writerows(testm["per_class"])
    result = dict(method=method,seed=seed,best_epoch=best_epoch,completed_epochs=current,
                  target_epochs=epochs,complete=current>=epochs,selected_by="max validation ECC",
                  test=testm,heat_filter=dict(selected_t=chosen, val_ecc=val_scores, test=heat_scores),
                  train_seconds=elapsed+time.time()-run_start)
    atomic_json(final_path, result)
    print(f"[EVALUATED] {method} seed{seed}: ACC {testm['acc']:.3f} ECC {testm['ecc']:.3f} "
          f"BEST={best_epoch}, COMPLETE={current>=epochs}",flush=True)
    del load, model
    torch.cuda.empty_cache()


def preflight(args):
    EXP.mkdir(parents=True,exist_ok=True)
    classes=load_classes()
    manifest=read_json(ROOT/"manifest.json")
    assert manifest and manifest.get("source_shards",0)>=20, "Full EmoSet manifest missing: run the full data preparation cell."
    print("Repo:",REPO,"\nFull EmoSet:",ROOT,"\nClasses:",classes,flush=True)
    for split in ("train","val","test"):
        ds=FullEmotionDataset(split,classes)
        print(f"Full {split}: {len(ds):,} images",flush=True)
        assert len(ds)>1000, "Tiny dataset detected; refused to run an allegedly full-data experiment"
    if torch.cuda.device_count()!=2:raise RuntimeError(f"Expected T4 x2, found {torch.cuda.device_count()} GPUs")
    for i in range(2):print("GPU",i,torch.cuda.get_device_name(i),flush=True)
    # Download weights once before starting parallel GPU workers.
    backbone=resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    del backbone
    atomic_json(EXP/"dataset_manifest.json",manifest)
    create_cache(classes)


def measure_epoch(config):
    """Short warmup benchmark on the ACTUAL full dataset; scales to epoch cost."""
    classes=load_classes()
    torch.set_num_threads(2)
    ds=FullEmotionDataset("train",classes,train=True)
    va=FullEmotionDataset("val",classes,train=False)
    workers=config["num_workers"] if ds.array is not None else max(config["num_workers"],2)
    dl=DataLoader(ds,batch_size=config["batch_size"],shuffle=True,num_workers=workers)
    vd=DataLoader(va,batch_size=config["batch_size"],num_workers=workers)
    W=build_cost_matrix(classes);rank=build_rank_matrix(W)
    model=model_init(12345)
    criterion=build_loss(METHODS["graph_a1p0"],W,rank).cuda()
    optimizer=torch.optim.AdamW(model.parameters(),lr=config["lr"],weight_decay=config["weight_decay"])
    scaler=torch.amp.GradScaler("cuda",enabled=config["amp"])
    measured=[]
    dl_iter=iter(dl)
    for i in range(min(48,len(dl))):
        t=time.perf_counter()
        x,y=next(dl_iter)
        x=x.cuda(non_blocking=True);y=y.cuda(non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda",enabled=config["amp"]): logits=model(x)
        with torch.autocast(device_type="cuda",enabled=False): loss=criterion(logits.float(),y)
        scaler.scale(loss).backward();scaler.step(optimizer);scaler.update()
        torch.cuda.synchronize()
        if i>=8:measured.append(time.perf_counter()-t)
    assert len(measured)>10
    tr_batch=float(np.median(measured))
    model.eval();vt=[]
    with torch.no_grad():
        vd_iter=iter(vd)
        for i in range(min(16,len(vd))):
            t=time.perf_counter()
            x,y=next(vd_iter)
            with torch.autocast(device_type="cuda",enabled=config["amp"]):model(x.cuda(non_blocking=True))
            torch.cuda.synchronize()
            if i>=2:vt.append(time.perf_counter()-t)
    va_batch=float(np.median(vt))
    # Add conservative contention and epoch checkpoint overhead: both GPUs use 4 CPU cores.
    expected=(tr_batch*math.ceil(len(ds)/config["batch_size"])+
              va_batch*math.ceil(len(va)/config["batch_size"]))*1.75 + 25
    del model,optimizer,criterion,scaler,dl,vd
    torch.cuda.empty_cache()
    return dict(median_train_batch_sec=tr_batch,median_val_batch_sec=va_batch,
                estimated_parallel_epoch_sec=expected,train_count=len(ds),val_count=len(va))


def worker(args):
    assert torch.cuda.device_count()==1, "Each worker must see exactly one assigned GPU"
    torch.set_num_threads(3)
    config=read_json(EXP/"experiment_config.json")
    jobs=read_json(args.plan)
    for method,seed in jobs:
        if time.time() >= args.deadline - config["reserve_seconds"]:
            print("[BUDGET] No time to start another run",flush=True)
            break
        try:
            training_job(method,int(seed),config,args.deadline)
        except Exception:
            err=traceback.format_exc()
            dest=RUNS/f"{method}_seed{seed}"
            dest.mkdir(parents=True,exist_ok=True)
            (dest/"error.txt").write_text(err)
            print(err,flush=True)
            raise


def orchestrate(args):
    from subprocess import Popen
    config=read_json(EXP/"experiment_config.json")
    preflight(args)
    benchmark=measure_epoch(config)
    deadline=float(config["deadline_ts"])
    available=deadline-time.time()-config["reserve_seconds"]
    # Schedule all coefficients and seeds across two GPUs; plan for the longer lane.
    jobs_per_lane=math.ceil(len(config["methods"])*len(config["seeds"])/2)
    safe_epoch=benchmark["estimated_parallel_epoch_sec"]
    capacity=max(0,math.floor(available/max(1,safe_epoch*jobs_per_lane)))
    target=min(config["max_epochs"],capacity)
    config["epochs"]=target
    benchmark.update(dict(available_training_hours=available/3600,
                          jobs_per_gpu_max=jobs_per_lane,capacity_epochs=capacity,
                          chosen_epochs=target,full_budget_hours=config["max_hours"]))
    atomic_json(EXP/"benchmark_and_epoch_plan.json",benchmark)
    atomic_json(EXP/"experiment_config.json",config)
    print("RUNTIME BENCHMARK:",json.dumps(benchmark,indent=2),flush=True)
    if target<config["min_epochs"]:
        raise RuntimeError(f"Can fit only {target} epochs/run with a safety margin; minimum requested is {config['min_epochs']}. Prepare data in a separate session or change MIN_EPOCHS explicitly.")
    print(f"Running {target} epochs per model Ã— {len(config['methods'])} methods Ã— {len(config['seeds'])} seeds, with two parallel T4 workers.",flush=True)
    jobs=[(m,int(seed)) for seed in config["seeds"] for m in config["methods"]]
    lanes=[jobs[::2],jobs[1::2]]
    for i in range(2):atomic_json(EXP/f"gpu{i}_jobs.json",lanes[i])
    processes=[]
    for i in range(2):
        log=open(EXP/f"gpu{i}.log","a",buffering=1)
        env=os.environ.copy();env["CUDA_VISIBLE_DEVICES"]=str(i)
        env["GSP_REPO"]=str(REPO);env["GSP_OUTPUT"]=str(EXP)
        cmd=[sys.executable,str(Path(__file__).resolve()),"worker","--plan",str(EXP/f"gpu{i}_jobs.json"),"--deadline",str(deadline)]
        proc=Popen(cmd,cwd=REPO,env=env,stdout=log,stderr=log)
        processes.append((proc,log))
    print("GPU0 jobs:",lanes[0],"\nGPU1 jobs:",lanes[1],flush=True)
    print("Two independent GPU workers launched.",flush=True)
    while any(p.poll() is None for p,_ in processes):
        time.sleep(30)
        status=[]
        for i,(proc,_) in enumerate(processes):
            try:lines=(EXP/f"gpu{i}.log").read_text(errors="replace").splitlines()
            except OSError:lines=[]
            status.append(f"GPU{i}: {lines[-1][:200] if lines else 'starting'}")
        print(" | ".join(status),flush=True)
    for p,lg in processes:lg.close()
    bad=[(i,p.returncode) for i,(p,_) in enumerate(processes) if p.returncode]
    if bad:print("Worker failure(s):",bad,"review gpu*.log",flush=True)
    report(args)
    if bad:raise RuntimeError(f"GPU workers failed: {bad}; a partial output.zip was assembled")


def group_metrics(results,methods):
    summary={}
    for method in methods:
        rr=[r for r in results if r["method"]==method]
        if not rr:continue
        summary[method]={"runs":len(rr),"complete_runs":sum(r["complete"] for r in rr)}
        for metric in ("acc","ecc","emc","acc2","macro_f1","gsee","brier","ece","n_errors"):
            vals=[float(r["test"][metric]) for r in rr]
            summary[method][metric] = dict(mean=float(np.mean(vals)),
                                           std=float(np.std(vals,ddof=1)) if len(vals)>1 else 0.0)
    return summary


def validation_only_selection(results, methods):
    """Choose alpha using validation ECC ONLY, never test scores."""
    scores={}
    for method in methods:
        run_scores=[]
        for r in results:
            if r["method"] != method:continue
            h=read_json(RUNS / f"{method}_seed{r['seed']}" / "history.json",[])
            if h:run_scores.append(max(float(t["val_ecc"]) for t in h))
        if run_scores:
            scores[method]=dict(mean_best_val_ecc=float(np.mean(run_scores)),
                                std_best_val_ecc=float(np.std(run_scores,ddof=1)) if len(run_scores)>1 else 0.0,
                                seed_count=len(run_scores),alpha=float(METHODS[method]["alpha"]))
    graph_methods=[m for m in methods if m != "ce" and m in scores]
    # Prefer smaller alpha if the mean validation scores tie.
    selected=max(graph_methods,key=lambda m:(scores[m]["mean_best_val_ecc"],-METHODS[m]["alpha"])) if graph_methods else None
    return dict(selected_graph_method=selected,
                selected_alpha=float(METHODS[selected]["alpha"]) if selected else None,
                baseline="ce",
                rule="Best mean across three seeds of each run's maximum validation ECC; test data never enters the selection rule.",
                validation_scores=scores,
                note="Selecting among coefficients via validation is distinct from test evaluation; full test metrics for every alpha are exploratory." )


def bootstrap_differences(results,method,metric="ecc",repeats=450):
    """Paired, class-stratified test-image CI conditional on fitted model seeds."""
    completed={(r["method"],r["seed"]) for r in results}
    seeds=sorted(s for s in set(r["seed"] for r in results) if ("ce",s) in completed and (method,s) in completed)
    if not seeds:return None
    pairs=[]
    for seed in seeds:
        with np.load(RUNS/f"ce_seed{seed}"/"predictions.npz") as aa, np.load(RUNS/f"{method}_seed{seed}"/"predictions.npz") as bb:
            y=aa["labels"].copy()
            assert np.array_equal(y,bb["labels"])
            pairs.append((y,aa["probs"].argmax(1),bb["probs"].argmax(1)))
    classes=load_classes();W=build_cost_matrix(classes);pol=polarity_vector(classes)
    y=pairs[0][0]
    assert all(np.array_equal(y,py) for py,_,_ in pairs)
    rng=np.random.default_rng(2206)
    ix=[np.where(y==c)[0] for c in range(len(classes))]
    draws=[]
    for k in range(repeats+1):
        idx=np.arange(len(y)) if k==0 else np.concatenate([rng.choice(a,size=len(a),replace=True) for a in ix])
        diffs=[evaluate_predictions(y[idx],b[idx],W,pol)[metric]-evaluate_predictions(y[idx],a[idx],W,pol)[metric] for _,a,b in pairs]
        draws.append(float(np.mean(diffs)))
    return dict(method=method,baseline="ce",metric=metric,seed_pairs=seeds,
                point_estimate=draws[0],ci95=[float(x) for x in np.quantile(draws[1:],[0.025,0.975])],
                definition="Paired class-stratified image bootstrap, conditional on these trained models and held-out test images. Does not estimate variability over future training seeds.")


def savefig(name):
    import matplotlib.pyplot as plt
    PLOTS.mkdir(parents=True,exist_ok=True)
    plt.savefig(PLOTS/name,dpi=200,bbox_inches="tight")
    plt.close()


def plot_results(results,summary,selection):
    import matplotlib.pyplot as plt
    methods=[m for m in METHODS if m in summary]
    labels={m:display_name(m) for m in methods}
    vals=[(float(METHODS[m]["alpha"]),m) for m in methods]
    fig,axes=plt.subplots(2,3,figsize=(16,9))
    for ax,(metric,title) in zip(axes.flat,[ ("acc","Accuracy (%) â†‘"),("ecc","ECC (%) â†‘"),
                                              ("emc","EMC (%) â†‘"),("acc2","Polarity ACC2 (%) â†‘"),
                                              ("macro_f1","Macro F1 (%) â†‘"),("gsee","GSEE â†“")]):
        xx=np.arange(len(methods))
        yy=[summary[m][metric]["mean"] for m in methods]
        err=[summary[m][metric]["std"] for m in methods]
        bars=ax.bar(xx,yy,yerr=err,capsize=3,alpha=.85)
        for j,bar in enumerate(bars):
            if methods[j]==selection.get("selected_graph_method"):bar.set_edgecolor("black");bar.set_linewidth(2)
        ax.set_xticks(xx,[labels[m] for m in methods],rotation=20,ha="right")
        ax.set_title(title);ax.grid(axis="y",alpha=.2)
        for bar,v in zip(bars,yy):ax.text(bar.get_x()+bar.get_width()/2, v, f"{v:.2f}",ha="center",va="bottom",fontsize=8)
    fig.suptitle("FULL EmoSet: test metrics across alpha values (mean Â± sample SD)",fontsize=15)
    fig.tight_layout();savefig("01_test_metrics_alpha_sweep.png")

    fig,axes=plt.subplots(1,3,figsize=(17,5.5))
    for ax,(field,title) in zip(axes,[("train_loss","Training loss (loss scales differ)"),
                                       ("val_acc","Validation accuracy (%)"),("val_ecc","Validation ECC (%)")]):
        for method in methods:
            histories=[read_json(RUNS/f"{method}_seed{r['seed']}"/"history.json",[]) for r in results if r["method"]==method]
            if not histories:continue
            n=min(len(h) for h in histories)
            if n<=0:continue
            a=np.array([[float(t[field]) for t in h[:n]] for h in histories])
            line=ax.plot(np.arange(1,n+1),a.mean(0),label=labels[method])[0]
            ax.fill_between(np.arange(1,n+1),a.mean(0)-a.std(0),a.mean(0)+a.std(0),color=line.get_color(),alpha=.12)
        ax.set_title(title);ax.set_xlabel("Epoch");ax.grid(alpha=.2)
    axes[2].legend(fontsize=8)
    fig.suptitle("Learning curves over 3 seeds â€” compare validation metrics, not raw loss magnitudes",fontsize=13)
    fig.tight_layout();savefig("02_learning_curves.png")

    fig,axes=plt.subplots(1,2,figsize=(13,5.2))
    xx=[x for x,_ in vals]
    for ax,metric,title in [(axes[0],"ecc","Test ECC vs coefficient Î±"),(axes[1],"acc","Test ACC vs coefficient Î±")]:
        yy=[summary[m][metric]["mean"] for _,m in vals]
        ee=[summary[m][metric]["std"] for _,m in vals]
        ax.errorbar(xx, yy, yerr=ee,fmt="o-",capsize=4)
        ax.set_xlabel("Graph coefficient Î± (0 = CE)");ax.set_ylabel(metric.upper()+" (%)");ax.set_title(title)
        ax.grid(alpha=.2)
        selected=selection.get("selected_alpha")
        if selected is not None:ax.axvline(selected,linestyle="--",alpha=.5,label=f"Validation-selected Î±={selected:g}");ax.legend()
    fig.suptitle("Coefficient sensitivity on held-out test (exploratory; alpha selected on validation)")
    fig.tight_layout();savefig("03_alpha_test_sensitivity.png")

    fig,ax=plt.subplots(figsize=(8.4,5))
    val=selection.get("validation_scores",{})
    mx=[METHODS[m]["alpha"] for m in methods if m in val]
    my=[val[m]["mean_best_val_ecc"] for m in methods if m in val]
    me=[val[m]["std_best_val_ecc"] for m in methods if m in val]
    ax.errorbar(mx,my,yerr=me,fmt="o-",capsize=4)
    ax.set_xlabel("Graph coefficient Î± (0 = CE)");ax.set_ylabel("Mean best validation ECC (%)")
    ax.set_title("Validation-only coefficient selection (across seeds)");ax.grid(alpha=.2)
    if selection.get("selected_alpha") is not None:ax.axvline(selection["selected_alpha"],linestyle="--",alpha=.5)
    fig.tight_layout();savefig("04_validation_alpha_selection.png")

    classes=load_classes()
    # One confusion matrix for baseline, every alpha and selected graph.
    for m in methods:
        cms=[np.loadtxt(RUNS/f"{m}_seed{r['seed']}"/"confusion_matrix.csv",delimiter=",") for r in results if r["method"]==m]
        cm=np.sum(cms,axis=0)
        norm=cm/np.maximum(cm.sum(1,keepdims=True),1)
        fig,ax=plt.subplots(figsize=(9,7))
        img=ax.imshow(norm,vmin=0,vmax=1,cmap="Blues")
        ax.set_xticks(np.arange(8),classes,rotation=40,ha="right")
        ax.set_yticks(np.arange(8),classes)
        ax.set_xlabel("Predicted emotion");ax.set_ylabel("True emotion")
        ax.set_title(f"{labels[m]} â€” pooled normalized test confusion")
        for i in range(8):
            for j in range(8):
                ax.text(j,i,f"{100*norm[i,j]:.0f}%",ha="center",va="center",fontsize=8,
                        color="white" if norm[i,j]>.55 else "black")
        fig.colorbar(img,ax=ax,label="Fraction of true class")
        fig.tight_layout();savefig(f"05_confusion_{m}.png")

    fig,ax=plt.subplots(figsize=(14,6))
    x=np.arange(len(classes));width=.8/len(methods)
    for j,m in enumerate(methods):
        per=np.array([[z["recall"] for z in r["test"]["per_class"]] for r in results if r["method"]==m])
        ax.bar(x-.4+width*(j+.5),per.mean(0),width,label=labels[m])
    ax.set_xticks(x,classes,rotation=30,ha="right");ax.set_ylim(0,100);ax.set_ylabel("Recall (%)")
    ax.set_title("Per-emotion recall across coefficients");ax.legend(fontsize=8);ax.grid(axis="y",alpha=.2)
    fig.tight_layout();savefig("06_per_emotion_recall.png")

    _,L,eigenvalues,U=graph_basis(build_cost_matrix(classes))
    fig,axes=plt.subplots(1,2,figsize=(14,5))
    axes[0].plot(range(len(eigenvalues)),eigenvalues,"o-")
    axes[0].set_title("Emotion-label Laplacian frequencies");axes[0].set_xlabel("Graph mode")
    axes[0].set_ylabel("Eigenvalue Î»")
    for m in methods:
        en=np.array([r["test"]["error_energy"] for r in results if r["method"]==m])
        axes[1].plot(range(8),en.mean(0),"o-",label=labels[m])
    axes[1].set_title("Test prediction error energy across graph modes");axes[1].set_xlabel("Graph mode")
    axes[1].set_ylabel("Mean squared residual coefficient");axes[1].legend(fontsize=8)
    for ax in axes:ax.grid(alpha=.2)
    fig.tight_layout();savefig("07_graph_frequency_error.png")

    fig,ax=plt.subplots(figsize=(7.5,6))
    ax.plot([0,1],[0,1],"--",color="gray",label="Perfectly calibrated")
    for m in methods:
        bins=defaultdict(lambda:[0,0.,0.])
        for r in results:
            if r["method"]!=m:continue
            for j,t in enumerate(r["test"]["calibration"]):
                if t["count"]:
                    bins[j][0]+=t["count"]
                    bins[j][1]+=t["count"]*t["accuracy"]
                    bins[j][2]+=t["count"]*t["confidence"]
        ix=sorted(bins)
        ax.plot([bins[j][2]/bins[j][0] for j in ix],[bins[j][1]/bins[j][0] for j in ix],"o-",label=labels[m])
    ax.set_xlabel("Mean confidence");ax.set_ylabel("Observed accuracy");ax.set_title("Test reliability diagram")
    ax.legend(fontsize=8);ax.grid(alpha=.2);fig.tight_layout();savefig("08_calibration.png")


def gallery(results,selection):
    import matplotlib.pyplot as plt
    chosen=selection.get("selected_graph_method")
    if not chosen:return
    available={(r["method"],r["seed"]) for r in results}
    common=sorted(s for m,s in available if m=="ce" and (chosen,s) in available)
    if not common:return
    seed=common[0]
    classes=load_classes();ds=FullEmotionDataset("test",classes)
    with np.load(RUNS/f"ce_seed{seed}"/"predictions.npz") as aa, np.load(RUNS/f"{chosen}_seed{seed}"/"predictions.npz") as bb:
        pce=aa["probs"].copy();pg=bb["probs"].copy()
    y=ds.labels;ce=pce.argmax(1);graph=pg.argmax(1)
    cases=[("Graph correct, CE wrong",np.flatnonzero((ce!=y)&(graph==y))),
           ("CE correct, Graph wrong",np.flatnonzero((ce==y)&(graph!=y))),
           ("Both wrong",np.flatnonzero((ce!=y)&(graph!=y))),
           ("Both correct",np.flatnonzero((ce==y)&(graph==y)))]
    rng=np.random.default_rng(917)
    examples=[(name,int(i)) for name,ids in cases for i in (rng.choice(ids,size=min(3,len(ids)),replace=False) if len(ids) else [])]
    if not examples:return
    fig,axs=plt.subplots(math.ceil(len(examples)/3),3,figsize=(15,4.1*math.ceil(len(examples)/3)))
    axs=np.array(axs).reshape(-1)
    for ax,(name,i) in zip(axs,examples):
        with Image.open(ds.paths[i]) as im:
            rgb=np.asarray(im.convert("RGB"))[16:176,16:176]
        ax.imshow(rgb);ax.axis("off")
        ax.set_title(f"{name}\nTruth: {classes[y[i]]}\nCE: {classes[ce[i]]} / Î±={selection['selected_alpha']:g}: {classes[graph[i]]}",fontsize=10)
    for ax in axs[len(examples):]:ax.axis("off")
    fig.suptitle(f"Qualitative examples â€” validation-selected Î±={selection['selected_alpha']:g}, seed {seed}")
    fig.tight_layout();savefig("09_prediction_examples.png")


def report(args):
    EXP.mkdir(parents=True,exist_ok=True);PLOTS.mkdir(parents=True,exist_ok=True)
    config=read_json(EXP/"experiment_config.json",{})
    methods=config.get("methods",list(METHODS))
    results=[]
    for path in sorted(RUNS.glob("*/metrics.json")):
        r=read_json(path)
        if r and r.get("method") in methods and r.get("seed") in config.get("seeds",[]):results.append(r)
    summary=group_metrics(results,methods)
    atomic_json(EXP/"summary.json",summary)
    selection=validation_only_selection(results,methods)
    atomic_json(EXP/"validation_alpha_selection.json",selection)
    with (EXP/"summary.csv").open("w",newline="") as f:
        cols=["method","alpha","runs","complete_runs","acc_mean","acc_std","ecc_mean","ecc_std", "emc_mean","emc_std","acc2_mean","macro_f1_mean","gsee_mean","ece_mean"]
        writer=csv.DictWriter(f,fieldnames=cols);writer.writeheader()
        for m,row in summary.items():
            writer.writerow(dict(method=m,alpha=METHODS[m]["alpha"],runs=row["runs"],complete_runs=row["complete_runs"],
                **{f"{k}_{stat}":row[k][stat] for k,stat in [("acc","mean"),("acc","std"),("ecc","mean"),("ecc","std"),("emc","mean"),("emc","std"),("acc2","mean"),("macro_f1","mean"),("gsee","mean"),("ece","mean")]}))
    comparisons={}
    chosen=selection.get("selected_graph_method")
    if chosen:
        comparisons={metric:bootstrap_differences(results,chosen,metric=metric) for metric in ("acc","ecc")}
    atomic_json(EXP/"selected_alpha_vs_ce_bootstrap.json",comparisons)
    if results:
        plot_results(results,summary,selection)
        gallery(results,selection)
    rows=[]
    for m in methods:
        if m not in summary:continue
        z=summary[m]
        rows.append(f"| {display_name(m)} | {z['runs']} | {z['complete_runs']} | {z['acc']['mean']:.3f} Â± {z['acc']['std']:.3f} | {z['ecc']['mean']:.3f} Â± {z['ecc']['std']:.3f} | {z['emc']['mean']:.3f} Â± {z['emc']['std']:.3f} | {z['gsee']['mean']:.4f} Â± {z['gsee']['std']:.4f} |")
    manifest=read_json(EXP/"dataset_manifest.json",{})
    md=["# Full EmoSet: 8-epoch graph loss coefficient sweep", "",
        "- Alpha values: 0 (CE), 0.1, 0.3, 0.5, 1.0. Gamma=1, beta=0.01 for graph runs.",
        f"- Seeds: {config.get('seeds')}; target epochs: {config.get('epochs')}; two independent T4 workers.",
        "- Every run selects its checkpoint by maximum validation ECC.",
        "- Global alpha selected ONLY from each run's best validation ECC averaged across seeds.",
        "- Held-out test comparisons for non-selected alpha values are exploratory.",
        f"- Dataset images: train {manifest.get('train',{}).get('images','?')}, validation {manifest.get('val',{}).get('images','?')}, test {manifest.get('test',{}).get('images','?')}.",
        "", "## Validation selection", "",
        f"Chosen graph coefficient: **Î±={selection.get('selected_alpha')}** ({selection.get('selected_graph_method')})", "",
        "| Alpha | Mean best validation ECC | SD | Runs |", "|---:|---:|---:|---:|" ]
    for m,info in selection["validation_scores"].items():
        md.append(f"| {info['alpha']:g} | {info['mean_best_val_ecc']:.3f} | {info['std_best_val_ecc']:.3f} | {info['seed_count']} |")
    md.extend(["", "## Held-out test scores (mean Â± sample SD)", "",
               "| Method | Runs | Complete | Accuracy % | ECC % | EMC % | GSEE â†“ |",
               "|---|---:|---:|---:|---:|---:|---:|",*rows, "",
               "## Selected graph alpha vs CE â€” paired test-image bootstrap", ""])
    for metric,obj in comparisons.items():
        if obj:md.append(f"- {metric.upper()}: {obj['point_estimate']:+.3f} percentage points, conditional 95% interval [{obj['ci95'][0]:+.3f}, {obj['ci95'][1]:+.3f}].")
    md.extend(["", "## Outputs", "",
               "- `runs/*/best.pt`: best validation ECC checkpoint for each coefficient/seed.",
               "- `runs/*/history.json`: per-epoch loss and validation metrics.",
               "- `runs/*/metrics.json`: test metrics, calibration, per-emotion performance.",
               "- `runs/*/predictions.npz`: test probabilities and labels.",
               "- `plots/`: comparison bars, coefficient sweep, validation selection, learning curves, confusion matrices, GFT diagnostics, qualitative samples.",
               "- `validation_alpha_selection.json`: selected alpha and validation results (no test scores in selection).",
               "- `selected_alpha_vs_ce_bootstrap.json`: paired conditional confidence intervals.",
               "- `dataset_manifest.json`: source shard provenance and actual full-data class counts.", "",
               "## Limitations", "",
               "- Low epoch budget (8) selected to match earlier validation plateau; confirm on independent repetitions.",
               "- Models all have the same dataset, optimizer, pretrained weights and training budget; loss magnitudes are not comparable across objectives.",
               "- Alpha selection is based on the validation set, but the selection and checkpointing introduce validation optimism; test remains held out.",
               "- Test scores for all other alpha values are exploratory and should not drive further hyperparameter selection without another held-out evaluation.",
               "- Three seeds do not establish generality across architectures, populations, and dataset mirrors.",
               "- No dataset photos in ZIP. Resume checkpoints stay in /kaggle/working output directory, not in ZIP."])
    (EXP/"README_RESULTS.md").write_text("\n".join(md)+"\n",encoding="utf8")
    archive=Path("/kaggle/working/output.zip") if Path("/kaggle/working").exists() else Path("/mnt/data/output.zip")
    with zipfile.ZipFile(archive,"w",compression=zipfile.ZIP_STORED,allowZip64=True) as zf:
        for f in sorted(EXP.rglob("*")):
            if not f.is_file() or f.name=="last_resume.pt" or f.suffix==".tmp":continue
            zf.write(f,f.relative_to(EXP))
        zf.write(Path(__file__).resolve(),"code/alpha_sweep_dual_t4_runner.py")
    print("\n=========== COEFFICIENT SWEEP REPORT ===========",flush=True)
    print("Runs evaluated:",len(results),"of",len(methods)*len(config.get("seeds",[])),flush=True)
    print("Validation-selected graph alpha:",selection.get("selected_alpha"),flush=True)
    for k,v in summary.items():print(f"{display_name(k)}: ACC {v['acc']['mean']:.3f} ECC {v['ecc']['mean']:.3f}; {v['runs']} seeds",flush=True)
    print("ZIP:",archive,"MiB:",round(archive.stat().st_size/1024**2,1),flush=True)


def main():
    parser=argparse.ArgumentParser()
    subs=parser.add_subparsers(dest="command",required=True)
    subs.add_parser("orchestrate")
    worker_parser=subs.add_parser("worker")
    worker_parser.add_argument("--plan",required=True)
    worker_parser.add_argument("--deadline",required=True,type=float)
    subs.add_parser("report")
    args=parser.parse_args()
    if args.command=="worker":worker(args)
    elif args.command=="orchestrate":orchestrate(args)
    else:report(args)

if __name__=="__main__":main()
