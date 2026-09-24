"""Pseudo-cutoff self-supervision: derive labels from observed continuation at earlier cutoffs."""
import sys, numpy as np, pandas as pd, lightgbm as lgb, pickle, os, argparse, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from streams import *; from features import build_features, assign_family; from pairwise import to_long
from protocol import *

def pseudo_labels(full, PC, horizon=90, min_prior=2):
    """full: all transactions (day relative to true cutoff). Label at pseudo-cutoff PC."""
    cand = full[(full.direction=='out') & (full.type=='card_payment') & (~full.isnoise) & (~full.isnonrec)].sort_values(['client_id','day'])
    labels = {}
    for cid, g in cand.groupby('client_id', sort=False):
        fam = assign_family(g).values
        d = g.day.values
        before = d < PC; fut = (d >= PC) & (d < PC + horizon)
        lab = 'none'
        for i in np.where(fut)[0]:
            f = fam[i]
            if f in FAM and (fam[before] == f).sum() >= min_prior:
                lab = f; break
        labels[cid] = lab
    return pd.Series(labels)

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--cutoffs', default='-90,-150,-210'); ap.add_argument('--out', default='pseudo.pkl')
    args = ap.parse_args()
    t0 = time.time()
    full = pd.concat([load('unlabeled_pretrain'), load('train'), load('valid'), load('test')], ignore_index=True)
    Xs = []; ys = []
    for PC in [int(x) for x in args.cutoffs.split(',')]:
        y = pseudo_labels(full, PC)
        hist = full[full.day < PC].copy(); hist['day'] = hist.day - PC; hist['ts'] = hist.ts - pd.Timedelta(days=-PC)
        X, _ = build_features(hist)
        y = y.reindex(X.index).fillna('none')
        X.index = [f'{c}@{PC}' for c in X.index]; y.index = X.index
        print('cutoff', PC, 'clients', len(X), 'label dist', y.value_counts(normalize=True).round(3).to_dict(), f'{time.time()-t0:.0f}s', flush=True)
        Xs.append(X); ys.append(y)
    Xp = pd.concat(Xs); yp = pd.concat(ys)
    pickle.dump((Xp, yp), open(args.out, 'wb'))
    print('saved', Xp.shape)
