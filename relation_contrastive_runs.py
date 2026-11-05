# ---------------------------------------------------------------------------
# Running the experiment: data preparation, one run, and several runs at the
# same time in separate worker processes. Every run is computed exactly as it
# would be on its own (same code, seed and data); only the scheduling differs.
# Small models like this one leave the GPU idle most of the time, so running
# a few trainings side by side shortens the total time.
# ---------------------------------------------------------------------------
import glob
import json
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch
from scipy import sparse

import relation_contrastive_method as rcm

MODE_OF = {"uniform": "uniform", "relation": "relation", "companions": "relation",
           "schane": "schane", "lcl": "lcl", "relation_schane": "relation_schane"}
HEARTBEAT_TIMEOUT = float(os.environ.get("RC_HEARTBEAT_TIMEOUT", 180))  # s without the notebook -> workers stop


def prepare_data(data, dev_frac, split_seed, related_pct, device):
    """Load the saved features and labels, split the training images
    (iterative stratification), scale the features (official StandardScaler,
    fitted on the training part) and build the relation types from the
    training part's labels. The notebook and every worker call this."""
    import context_reliance_check as crc
    IN, FEAT = os.path.join(data, "check_inputs"), os.path.join(data, "features")
    files = {"training labels": os.path.join(IN, "training_positives.npz"),
             "evaluation labels": os.path.join(IN, "val_positives.npz"),
             "label table": os.path.join(IN, "labels.csv"),
             "training features": os.path.join(FEAT, "train_features.npy"),
             "evaluation features": os.path.join(FEAT, "val_features.npy"),
             "text embeddings": os.path.join(FEAT, "label_text_embeddings.npy")}
    missing = [p for p in files.values() if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError("Missing (run coco_mlt_context_check.ipynb first): "
                                + ", ".join(missing))
    Y_all = sparse.load_npz(files["training labels"]).toarray().astype(np.float32)
    Y_test = np.load(files["evaluation labels"])["positives"].astype(np.float32)
    X_all = np.load(files["training features"]).astype(np.float32)
    X_test = np.load(files["evaluation features"]).astype(np.float32)
    labels_df = pd.read_csv(files["label table"])
    names = labels_df["display_name"].tolist()
    cohort = labels_df["cohort"].to_numpy()
    L = len(names)
    assert X_all.shape[0] == Y_all.shape[0] and X_test.shape[0] == Y_test.shape[0]
    assert Y_all.shape[1] == Y_test.shape[1] == L
    sim = crc.text_similarity(files["text embeddings"], None, L)
    split = rcm.iterative_split(Y_all, dev_frac, split_seed)       # 0 fit, 1 dev
    fit, dev = np.flatnonzero(split == 0), np.flatnonzero(split == 1)
    Y_fit, Y_dev = Y_all[fit], Y_all[dev]
    if (Y_fit.sum(0) == 0).any():
        raise ValueError("A class has no training image after the split.")
    scaler = rcm.Scaler().fit(X_all[fit])
    t = lambda a: torch.as_tensor(a, dtype=torch.float32, device=device)
    shared_fit = Y_fit.T.astype(np.int64) @ Y_fit.astype(np.int64)
    rel_seen, rel_never, informative, companion = rcm.relation_masks(
        sim, shared_fit, np.diag(shared_fit).copy(), len(fit), crc.related_labels,
        related_pct)
    return dict(Y_all=Y_all, Y_test=Y_test, names=names, cohort=cohort, L=L, sim=sim,
                fit=fit, dev=dev, Y_fit=Y_fit, Y_dev=Y_dev,
                Xf=t(scaler.transform(X_all[fit])), Xd=t(scaler.transform(X_all[dev])),
                Xt=t(scaler.transform(X_test)), Yf=t(Y_fit), Yd=t(Y_dev),
                rel_seen=rel_seen, rel_never=rel_never, informative=informative,
                companion=companion,
                RELATION=[(1.0, rel_seen), (0.5, rel_never), (0.25, informative)],
                COMPANION=[(1.0, companion)])


def pretrain_kwargs(kind, setting, D):
    relation = (D["RELATION"] if kind in ("relation", "relation_schane") else
                D["COMPANION"] if kind == "companions" else None)
    return dict(mode=MODE_OF[kind], relation=relation,
                alpha=setting.get("alpha", 0.0), tau=setting.get("tau"))


def dev_map(logits, Y_dev):
    import context_reliance_check as crc
    aps = [crc.average_precision(logits[:, j], Y_dev[:, j])
           for j in range(Y_dev.shape[1]) if Y_dev[:, j].any()]
    return 100 * float(np.mean(aps))


def run_spec(spec, D, pt_cfg, ft_cfg, device, report=None, epoch_hook=None):
    """One run: contrastive pretraining (contrastive kinds only), then
    fine-tuning. Returns the fields that are saved for the run."""
    kind, setting, seed = spec["kind"], spec["setting"], spec["seed"]
    t0 = time.time()
    history = np.zeros(0)
    ft = dict(ft_cfg, lrp=setting.get("lrp", ft_cfg["lrp"]))
    if kind in ("bce", "relbce"):
        w_fit = w_dev = None
        if kind == "relbce":
            tt = lambda a: torch.as_tensor(a, dtype=torch.float32, device=device)
            w_fit = tt(rcm.neg_weight_matrix(D["Y_fit"], D["RELATION"], setting["alpha"]))
            w_dev = tt(rcm.neg_weight_matrix(D["Y_dev"], D["RELATION"], setting["alpha"]))
        main, cap = rcm.finetune(D["Xf"], D["Yf"], D["Xd"], D["Yd"], D["Xt"], seed,
                                 device, ft, None, w_fit, w_dev, epoch_hook=epoch_hook)
    else:
        state, hist = rcm.pretrain(D["Xf"], D["Yf"], seed=seed, device=device,
                                   cfg=pt_cfg, report=report, epoch_hook=epoch_hook,
                                   **pretrain_kwargs(kind, setting, D))
        if spec.get("backbone_path"):
            torch.save(state, spec["backbone_path"])
        history = np.array(hist)
        main, cap = rcm.finetune(D["Xf"], D["Yf"], D["Xd"], D["Yd"], D["Xt"], seed,
                                 device, ft, backbone_state=state,
                                 epoch_hook=epoch_hook)
    return dict(test_logits=main["test_logits"], dev_logits=main["dev_logits"],
                best_epoch=main["best_epoch"], epochs_run=main["epochs_run"],
                stopped_early=main["stopped_early"], final_lr=main["final_lr"],
                dev_loss=main["dev_loss"], dev_map=dev_map(main["dev_logits"], D["Y_dev"]),
                test_logits_30=cap["test_logits"], best_epoch_30=cap["best_epoch"],
                dev_map_30=dev_map(cap["dev_logits"], D["Y_dev"]),
                history=history, seconds=time.time() - t0)


def save_run(path, out):
    """Write to a temporary file first, so a half-written run never counts."""
    tmp = path[:-len(".npz")] + f".tmp{os.getpid()}.npz"
    np.savez(tmp, **out)
    os.replace(tmp, path)


def load_run(path):
    with np.load(path) as f:
        return {k: f[k] for k in f.files}


def _sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


def _touch(path):
    with open(path, "a"):
        pass
    os.utime(path, None)


def _alive(heartbeat):
    try:
        return time.time() - os.path.getmtime(heartbeat) < HEARTBEAT_TIMEOUT
    except OSError:
        return False


class NotebookStopped(RuntimeError):
    pass


def worker_main(job_path, worker_id):
    """Entry point of a worker process: claims runs one at a time (a lock
    file per run), computes and saves them, and stops if the notebook stops."""
    with open(job_path, encoding="utf-8") as f:
        job = json.load(f)
    torch.set_num_threads(1)
    device, hb = job["device"], job["heartbeat"]
    D = prepare_data(job["data"], job["dev_frac"], job["split_seed"],
                     job["related_pct"], device)

    def hook(_epoch):
        if not _alive(hb):
            raise NotebookStopped("the notebook stopped; this worker exits")

    if job.get("benchmark"):
        b = job["benchmark"]
        cfg = dict(job["pt"], epochs=1, iters_per_epoch=20)
        kw = pretrain_kwargs("uniform", {}, D)
        rcm.pretrain(D["Xf"], D["Yf"], seed=0, device=device, cfg=cfg, **kw)  # warm-up
        _sync(device)
        open(os.path.join(b["dir"], f"ready{worker_id}"), "w").close()
        t_wait = time.time()                  # start timing together
        while (len(glob.glob(os.path.join(b["dir"], "ready*"))) < b["workers"]
               and time.time() - t_wait < 300):
            time.sleep(0.1)
        t0 = time.time()
        rcm.pretrain(D["Xf"], D["Yf"], seed=0, device=device,
                     cfg=dict(cfg, iters_per_epoch=b["iters"]), **kw)
        _sync(device)
        with open(os.path.join(b["dir"], f"result{worker_id}.json"), "w") as f:
            json.dump({"seconds_per_iter": (time.time() - t0) / b["iters"]}, f)
        return
    for spec in job["specs"]:
        if not _alive(hb):
            print("the notebook stopped; this worker exits", flush=True)
            return
        path = spec["path"]
        if os.path.exists(path):
            continue
        lock = path + ".lock"
        try:
            os.close(os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
        except FileExistsError:
            continue                          # another worker has this run
        try:
            print(f"[worker {worker_id}] started {spec['tag']}", flush=True)
            out = run_spec(spec, D, job["pt"], job["ft"], device, epoch_hook=hook)
            save_run(path, out)
            print(f"[worker {worker_id}] finished {spec['tag']} "
                  f"({out['seconds'] / 60:.1f} min)", flush=True)
        except NotebookStopped as e:
            print(str(e), flush=True)
            return
        finally:
            try:
                os.remove(lock)
            except OSError:
                pass


def _launch(job_path, n, log_dir, prefix):
    procs = []
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)      # no console windows on Windows
    for w in range(n):
        log = open(os.path.join(log_dir, f"{prefix}{w}.log"), "w", encoding="utf-8")
        code = (f"import relation_contrastive_runs as r; "
                f"r.worker_main({os.path.abspath(job_path)!r}, {w})")
        p = subprocess.Popen([sys.executable, "-u", "-c", code], stdout=log,
                             stderr=subprocess.STDOUT, cwd=os.getcwd(),
                             env=dict(os.environ, PYTHONIOENCODING="utf-8"),
                             creationflags=flags)
        procs.append((p, log))
    return procs


def _stop(procs):
    for p, _ in procs:
        if p.poll() is None:
            p.terminate()
    for p, log in procs:
        try:
            p.wait(timeout=15)
        except subprocess.TimeoutExpired:
            p.kill()
        log.close()


def _log_tail(log_dir, prefix, w, n=25):
    try:
        with open(os.path.join(log_dir, f"{prefix}{w}.log"), encoding="utf-8",
                  errors="replace") as f:
            return "".join(f.readlines()[-n:])
    except OSError:
        return "(no log)"


def benchmark_parallel(n, job, out_dir, iters=300, timeout=900):
    """Seconds per pretraining iteration when n workers train at the same
    time (uniform loss), or None if the workers could not run."""
    log_dir = os.path.join(out_dir, "logs")
    bdir = os.path.join(log_dir, "benchmark")
    os.makedirs(bdir, exist_ok=True)
    for f in glob.glob(os.path.join(bdir, "*")):
        os.remove(f)
    job = dict(job, specs=[], benchmark=dict(workers=n, dir=os.path.abspath(bdir),
                                             iters=iters))
    path = os.path.join(out_dir, "job_benchmark.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(job, f)
    _touch(job["heartbeat"])
    procs = _launch(path, n, log_dir, "benchmark_worker")
    t0 = time.time()
    try:
        while any(p.poll() is None for p, _ in procs) and time.time() - t0 < timeout:
            _touch(job["heartbeat"])
            time.sleep(1)
    finally:
        _stop(procs)
    res = []
    for w in range(n):
        try:
            with open(os.path.join(bdir, f"result{w}.json")) as f:
                res.append(json.load(f)["seconds_per_iter"])
        except OSError:
            print(f"Worker {w} did not finish the speed test:\n"
                  + _log_tail(log_dir, "benchmark_worker", w))
    return float(np.mean(res)) if len(res) == n else None


def available_ram_gb():
    """Free system memory in GB, or None if it cannot be read."""
    try:
        import psutil
        return psutil.virtual_memory().available / 2**30
    except Exception:
        pass
    if os.name == "nt":
        try:
            import ctypes

            class _MemStatus(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong),
                            ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong),
                            ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong),
                            ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

            ms = _MemStatus()
            ms.dwLength = ctypes.sizeof(_MemStatus)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms)):
                return ms.ullAvailPhys / 2**30
        except Exception:
            pass
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 2**20
    except OSError:
        pass
    return None


def auto_workers(device, ram_per_worker=1.2, vram_per_worker=0.75, cap=4):
    """How many trainings to run at once, limited by CPU cores, free memory
    and GPU memory. Returns (number, explanation)."""
    cores = os.cpu_count() or 2
    limits = {"CPU cores": max(1, cores // 2)}
    parts = [f"{cores} CPU threads"]
    ram = available_ram_gb()
    if ram is not None:
        limits["free memory"] = int((ram - 1.0) // ram_per_worker)
        parts.append(f"{ram:.1f} GB memory free")
    if device == "cuda":
        vram = torch.cuda.get_device_properties(0).total_memory / 2**30
        limits["GPU memory"] = int((vram - 1.0) // vram_per_worker)
        parts.append(f"{vram:.1f} GB GPU memory")
    n = max(1, min(cap, *limits.values()))
    text = ", ".join(parts) + f" -> up to {n} training(s) at once"
    if n < cap and limits.get("free memory", cap) == min(limits.values()):
        text += (" (limited by free memory: close other programs and other "
                 "notebooks' kernels, then rerun this cell, to allow more)")
    return n, text


def makespan(costs, n):
    """Finishing time when jobs (longest first) go to whichever of n workers
    is free first."""
    loads = [0.0] * max(1, n)
    for c in sorted(costs, reverse=True):
        loads[loads.index(min(loads))] += c
    return max(loads)


def execute(specs, workers, job, D, describe, cost, report=print):
    """Compute every run in specs whose result is not saved yet: one by one in
    this process (workers <= 1) or in parallel worker processes."""
    todo = [s for s in specs if not os.path.exists(s["path"])]
    if not todo:
        return
    todo.sort(key=lambda s: -cost[s["tag"]])       # longest first balances workers
    total = sum(cost[s["tag"]] for s in todo)
    state = dict(done=0.0, n=0, t0=time.time())

    def progress(s, out):
        state["done"] += cost[s["tag"]]
        state["n"] += 1
        el = time.time() - state["t0"]
        left = el / state["done"] * (total - state["done"]) if state["done"] else 0
        report(f"  [{state['n']}/{len(todo)}] {describe(s, out)}"
               + (f"; about {left / 60:.0f} min left in this phase"
                  if state["n"] < len(todo) else ""))

    if workers <= 1:
        for s in todo:
            out = run_spec(s, D, job["pt"], job["ft"], job["device"], report=report)
            save_run(s["path"], out)
            progress(s, out)
        return
    run_dir = os.path.dirname(todo[0]["path"])
    for f in (glob.glob(os.path.join(run_dir, "*.lock"))
              + glob.glob(os.path.join(run_dir, "*.tmp*.npz"))):
        os.remove(f)                       # left over from an interrupted session
    out_dir = os.path.dirname(run_dir)
    log_dir = os.path.join(out_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(out_dir, "job.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(dict(job, specs=todo), f)
    _touch(job["heartbeat"])
    n = min(workers, len(todo))
    procs = _launch(path, n, log_dir, "worker")
    reported = set()
    last_note = [time.time()]

    def sweep():
        for s in todo:
            if s["tag"] not in reported and os.path.exists(s["path"]):
                reported.add(s["tag"])
                last_note[0] = time.time()
                progress(s, load_run(s["path"]))

    try:
        while any(p.poll() is None for p, _ in procs):
            _touch(job["heartbeat"])
            sweep()
            if time.time() - last_note[0] > 600:
                last_note[0] = time.time()
                report(f"  ... still running ({(time.time() - state['t0']) / 60:.0f} min "
                       f"into this phase; {len(reported)} of {len(todo)} runs finished)")
            time.sleep(5)
        sweep()
    finally:
        _stop(procs)                       # also on an interrupt: no orphan workers
    missing = [s for s in todo if not os.path.exists(s["path"])]
    if missing:
        for w, (p, _) in enumerate(procs):
            if p.returncode:
                print(f"--- worker {w} stopped with an error; end of its log:\n"
                      + _log_tail(log_dir, "worker", w))
        raise RuntimeError(
            f"{len(missing)} run(s) did not finish (see above). Run this cell again "
            "to continue; finished runs are kept. If a worker ran out of GPU memory, "
            "set WORKERS lower in Step 1.")
