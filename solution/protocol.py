import hashlib, numpy as np, pandas as pd
from sklearn.metrics import f1_score
from streams import LABELS
def h(s): return int(hashlib.sha256(s.encode()).hexdigest(), 16)
def is_lockbox(cid): return h(cid) % 5 == 0
def fold_of(seed, cid): return h(f"{seed}:{cid}") % 5
def macro_f1(y, p): return f1_score(y, p, average='macro', labels=LABELS)
def per_class_f1(y, p): return dict(zip(LABELS, f1_score(y, p, average=None, labels=LABELS)))
def bootstrap_ci(y, p, n=1000, seed=0):
    rng = np.random.default_rng(seed); y=np.asarray(y); p=np.asarray(p); N=len(y); vals=[]
    for _ in range(n):
        idx = rng.integers(0, N, N); vals.append(macro_f1(y[idx], p[idx]))
    return np.percentile(vals, [2.5, 97.5])
