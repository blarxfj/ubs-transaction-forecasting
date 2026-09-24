"""Shared (client, family) binary model: one row per client x family, target = label==family."""
import sys, numpy as np, pandas as pd, lightgbm as lgb, pickle, os, argparse
sys.path.insert(0, 'sol')
from streams import *; from protocol import *

def to_long(X):
    """Convert wide client features to long (client, family) rows with family-agnostic column names."""
    fam_cols = {}
    for fm in FAM:
        fam_cols[fm] = [c for c in X.columns if c.startswith(fm+'_')]
    suffixes = sorted(set(c[len(fm)+1:] for fm in FAM for c in fam_cols[fm]))
    client_cols = [c for c in X.columns if not any(c.startswith(fm+'_') for fm in FAM)]
    parts = []
    for fi, fm in enumerate(FAM):
        cols = {}
        for s in suffixes:
            cols['f_'+s] = X[fm+'_'+s] if fm+'_'+s in X.columns else pd.Series(np.nan, index=X.index)
        for c in client_cols: cols[c] = X[c]
        P = pd.DataFrame(cols, index=X.index)
        # context: other families' strength
        others = [o for o in FAM if o != fm]
        on = X[[o+'_n' for o in others]].fillna(0)
        P['oth_max_n'] = on.max(1); P['oth_sum_n'] = on.sum(1); P['oth_n_active'] = (X[[o+'_last' for o in others]] > -60).sum(1)
        odue = X[[o+'_due' for o in others]].where(X[[o+'_last' for o in others]].values > -75)
        P['oth_min_due'] = odue.min(1); P['due_minus_oth_min'] = X[fm+'_due'] - odue.min(1) if fm+'_due' in X.columns else np.nan
        on90 = X[[o+'_n90' for o in others]].fillna(0); P['oth_max_n90'] = on90.max(1)
        P['fam_id'] = fi
        P['client_id'] = X.index; P['fam'] = fm
        parts.append(P)
    L = pd.concat(parts, ignore_index=True)
    return L

def decide(scores, thr):
    """scores: DataFrame index client, columns FAM. Return predicted label."""
    best = scores.idxmax(1); mx = scores.max(1)
    return np.where(mx > thr, best, 'none')

def best_threshold(scores, ytrue):
    best = (None, -1)
    for t in np.linspace(0.05, 0.6, 56):
        f = macro_f1(ytrue, decide(scores, t))
        if f > best[1]: best = (t, f)
    return best

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--feats', default='feats2.pkl'); ap.add_argument('--lr', type=float, default=0.03)
    ap.add_argument('--leaves', type=int, default=15); ap.add_argument('--minleaf', type=int, default=40); ap.add_argument('--ff', type=float, default=0.5); ap.add_argument('--l2', type=float, default=1.0)
    ap.add_argument('--seeds', default='0,1,2'); ap.add_argument('--save_oof', default=''); ap.add_argument('--drop', default=''); ap.add_argument('--extra', default='')
    args = ap.parse_args()
    lab = pd.concat([pd.read_csv('data/train_labels.csv'), pd.read_csv('data/valid_labels.csv')]).set_index('client_id').target_next_recurring_merchant
    X, A = pickle.load(open(args.feats,'rb'))
    if args.drop: X = X[[c for c in X.columns if not any(c.endswith(p) for p in args.drop.split(','))]]
    L = to_long(X)
    if args.extra:
        E = pickle.load(open(args.extra,'rb')); L['pseudo_s'] = [E.loc[c, f] if c in E.index else np.nan for c, f in zip(L.client_id, L.fam)]
    L['y'] = (L.client_id.map(lab) == L.fam).astype(float)
    feat_cols = [c for c in L.columns if c not in ('client_id','fam','y')]
    print('long shape', L.shape, 'features', len(feat_cols))
    dev = [c for c in lab.index if not is_lockbox(c)]; devset = set(dev)
    Ld = L[L.client_id.isin(devset)].reset_index(drop=True)
    valid_ids = set(pd.read_csv('data/valid_labels.csv').client_id)
    params = dict(objective='binary', learning_rate=args.lr, num_leaves=args.leaves, min_data_in_leaf=args.minleaf, feature_fraction=args.ff, bagging_fraction=0.8, bagging_freq=1, lambda_l2=args.l2, verbose=-1, num_threads=8)
    res=[]; resv=[]; oofs={}
    ytrue = lab.loc[dev]
    for seed in [int(s) for s in args.seeds.split(',')]:
        fold_map = {c: fold_of(seed, c) for c in dev}; folds = Ld.client_id.map(fold_map).values
        oof = np.zeros(len(Ld)); its=[]
        for k in range(5):
            trn = folds!=k; tst = folds==k
            m = lgb.train({**params, 'seed': seed}, lgb.Dataset(Ld.loc[trn, feat_cols], Ld.y[trn]), num_boost_round=3000, valid_sets=[lgb.Dataset(Ld.loc[tst, feat_cols], Ld.y[tst])], callbacks=[lgb.early_stopping(100, verbose=False)])
            oof[tst] = m.predict(Ld.loc[tst, feat_cols], num_iteration=m.best_iteration); its.append(m.best_iteration)
        S = pd.DataFrame({'client_id': Ld.client_id, 'fam': Ld.fam, 's': oof}).pivot(index='client_id', columns='fam', values='s').loc[dev, FAM]
        # threshold chosen per fold on the other folds (nested) to avoid optimism
        pred = np.empty(len(dev), dtype=object); fold_arr = np.array([fold_map[c] for c in dev])
        for k in range(5):
            t, _ = best_threshold(S[fold_arr!=k], ytrue[fold_arr!=k].values)
            pred[fold_arr==k] = decide(S[fold_arr==k], t)
        f = macro_f1(ytrue.values, pred); isv = np.array([c in valid_ids for c in dev]); fv = macro_f1(ytrue.values[isv], pred[isv])
        t_all, f_opt = best_threshold(S, ytrue.values)
        print('seed', seed, 'iters', its, 'macroF1(nested thr)', round(f,4), 'valid-only', round(fv,4), 'thr_all', round(t_all,3), 'f_opt', round(f_opt,4), {k: round(float(v),3) for k,v in per_class_f1(ytrue.values, pred).items()})
        res.append(f); resv.append(fv); oofs[seed] = S
    print('MEAN', round(np.mean(res),4), 'valid-only', round(np.mean(resv),4))
    if args.save_oof: pickle.dump((dev, ytrue, oofs), open(args.save_oof,'wb'))
