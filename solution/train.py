"""End-to-end pipeline: features -> per-family binary LightGBM (bagged) -> stage-2 calibration -> deliverables.

Outputs (in --out dir): oof.csv (seed 0), lockbox.csv, test_proba.csv, submission.csv, results.json
"""
import sys, os, json, pickle, argparse, time, hashlib
import numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from streams import *; from features import build_features; from pairwise import to_long
from stage2 import fit_stage2, predict_proba, decide_w
from protocol import *

PARAMS = dict(objective='binary', learning_rate=0.03, num_leaves=15, min_data_in_leaf=40, feature_fraction=0.5,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=8, deterministic=True, force_row_wise=True)

def fit_stage1(Ltr, feat_cols, seeds, n_rounds):
    models = []
    for s in seeds:
        m = lgb.train({**PARAMS, 'seed': s}, lgb.Dataset(Ltr[feat_cols], Ltr.y), num_boost_round=n_rounds)
        models.append(m)
    return models

def predict_stage1(models, L, feat_cols):
    s = np.mean([m.predict(L[feat_cols]) for m in models], axis=0)
    return pd.DataFrame({'client_id': L.client_id.values, 'fam': L.fam.values, 's': s}).pivot(index='client_id', columns='fam', values='s')[FAM]

def inner_oof(Ltr, feat_cols, seed, n_rounds, k=5):
    """stage-1 OOF scores on the training clients (for fitting stage 2)."""
    cids = Ltr.client_id.unique(); fm = {c: fold_of(1000+seed, c) for c in cids}
    folds = Ltr.client_id.map(fm).values; oof = np.zeros(len(Ltr))
    for j in range(k):
        m = lgb.train({**PARAMS, 'seed': seed}, lgb.Dataset(Ltr.loc[folds!=j, feat_cols], Ltr.y[folds!=j]), num_boost_round=n_rounds)
        oof[folds==j] = m.predict(Ltr.loc[folds==j, feat_cols])
    return pd.DataFrame({'client_id': Ltr.client_id.values, 'fam': Ltr.fam.values, 's': oof}).pivot(index='client_id', columns='fam', values='s')[FAM]

def fit_full(Ltr, feat_cols, seeds, n_rounds):
    """fit stage 1 (bagged over seeds) and stage 2 (on inner OOF of seed seeds[0])."""
    models = fit_stage1(Ltr, feat_cols, seeds, n_rounds)
    S_in = inner_oof(Ltr, feat_cols, seeds[0], n_rounds)
    y_in = Ltr.groupby('client_id').apply(lambda g: g.fam[g.y==1].iloc[0] if (g.y==1).any() else 'none').reindex(S_in.index)
    s2 = fit_stage2(S_in, y_in.values)
    return models, s2

def predict_full(models, s2, L, feat_cols):
    S = predict_stage1(models, L, feat_cols)
    return predict_proba(s2, S), S

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data'); ap.add_argument('--out', default='out'); ap.add_argument('--feats', default='')
    ap.add_argument('--extra', default='', help='optional pickle with extra per-(client,fam) features, e.g. pseudo scores')
    ap.add_argument('--rounds', type=int, default=450); ap.add_argument('--bag', type=int, default=5)
    ap.add_argument('--cv_seeds', default='0,1,2'); ap.add_argument('--skip_cv', action='store_true')
    args = ap.parse_args(); os.makedirs(args.out, exist_ok=True); t0 = time.time()
    os.chdir(os.path.dirname(os.path.abspath(args.data)) if os.path.isabs(args.data) else '.')
    lab = pd.concat([pd.read_csv(f'{args.data}/train_labels.csv'), pd.read_csv(f'{args.data}/valid_labels.csv')]).set_index('client_id').target_next_recurring_merchant
    valid_ids = set(pd.read_csv(f'{args.data}/valid_labels.csv').client_id)
    test_ids = pd.read_csv(f'{args.data}/sample_submission.csv').client_id.tolist()
    if args.feats and os.path.exists(args.feats):
        X, A = pickle.load(open(args.feats, 'rb'))
    else:
        tr = pd.concat([load('train'), load('valid'), load('test')])
        X, A = build_features(tr)
        if args.feats: pickle.dump((X, A), open(args.feats, 'wb'))
    print('features', X.shape, f'{time.time()-t0:.0f}s', flush=True)
    L = to_long(X)
    if args.extra:
        E = pickle.load(open(args.extra, 'rb'))  # DataFrame index client_id, columns FAM
        L['pseudo_s'] = [E.loc[c, f] if c in E.index else np.nan for c, f in zip(L.client_id, L.fam)]
    L['y'] = (L.client_id.map(lab) == L.fam).astype(float)
    feat_cols = [c for c in L.columns if c not in ('client_id', 'fam', 'y')]
    dev = [c for c in lab.index if not is_lockbox(c)]; lock = [c for c in lab.index if is_lockbox(c)]
    print('dev', len(dev), 'lockbox', len(lock), 'test', len(test_ids), 'features', len(feat_cols), flush=True)
    Ld = L[L.client_id.isin(set(dev))].reset_index(drop=True)
    results = {'n_dev': len(dev), 'n_lockbox': len(lock), 'n_test': len(test_ids), 'n_features': len(feat_cols), 'rounds': args.rounds, 'bag': args.bag}
    ydev = lab.loc[dev]
    # ---- protocol CV ----
    if not args.skip_cv:
        cv = {}
        for seed in [int(s) for s in args.cv_seeds.split(',')]:
            fm = {c: fold_of(seed, c) for c in dev}; folds = Ld.client_id.map(fm).values
            P = pd.DataFrame(0.0, index=pd.Index(dev), columns=LABELS); Sout = pd.DataFrame(0.0, index=pd.Index(dev), columns=FAM)
            for k in range(5):
                trn = Ld[folds != k].reset_index(drop=True); tst = Ld[folds == k].reset_index(drop=True)
                models, s2 = fit_full(trn, feat_cols, [seed], args.rounds)
                Pk, Sk = predict_full(models, s2, tst, feat_cols)
                P.loc[Pk.index] = Pk.values; Sout.loc[Sk.index] = Sk.values
                print(f'  seed {seed} fold {k} done {time.time()-t0:.0f}s', flush=True)
            pred = decide_w(P, np.ones(8)); y = ydev.values; isv = np.array([c in valid_ids for c in dev])
            cv[seed] = {'macro_f1': macro_f1(y, pred), 'macro_f1_valid_only': macro_f1(y[isv], pred[isv]), 'per_class': per_class_f1(y, pred),
                        'accuracy': float((y == pred).mean())}
            if seed == int(args.cv_seeds.split(',')[0]):
                ci = bootstrap_ci(y, pred); cv[seed]['bootstrap95'] = [float(ci[0]), float(ci[1])]
                oof = P.copy(); oof.insert(0, 'fold', [fm[c] for c in dev]); oof.index.name = 'client_id'
                oof.to_csv(f'{args.out}/oof.csv'); Sout.to_csv(f'{args.out}/oof_stage1_scores.csv')
            print('seed', seed, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in cv[seed].items() if k != 'per_class'}, flush=True)
        results['cv'] = cv; results['cv_mean_macro_f1'] = float(np.mean([v['macro_f1'] for v in cv.values()]))
        results['cv_mean_macro_f1_valid_only'] = float(np.mean([v['macro_f1_valid_only'] for v in cv.values()]))
        print('CV mean macro-F1', round(results['cv_mean_macro_f1'], 4), 'valid-only', round(results['cv_mean_macro_f1_valid_only'], 4), flush=True)
    # ---- final model on all dev clients ----
    models, s2 = fit_full(Ld, feat_cols, list(range(args.bag)), args.rounds)
    Lrest = L[~L.client_id.isin(set(dev))].reset_index(drop=True)
    P, S = predict_full(models, s2, Lrest, feat_cols)
    Plock = P.loc[lock]; Ptest = P.loc[test_ids]
    Plock.index.name = 'client_id'; Ptest.index.name = 'client_id'
    Plock.to_csv(f'{args.out}/lockbox.csv'); Ptest.to_csv(f'{args.out}/test_proba.csv')
    sub = pd.DataFrame({'client_id': test_ids, 'predicted_next_recurring_merchant': decide_w(Ptest, np.ones(8))})
    sub.to_csv(f'{args.out}/submission.csv', index=False)
    ylock = lab.loc[lock].values; plock = decide_w(Plock, np.ones(8))
    results['lockbox'] = {'macro_f1': macro_f1(ylock, plock), 'per_class': per_class_f1(ylock, plock), 'accuracy': float((ylock == plock).mean()),
                          'bootstrap95': [float(x) for x in bootstrap_ci(ylock, plock)]}
    results['submission_label_dist'] = sub.predicted_next_recurring_merchant.value_counts().to_dict()
    results['submission_sha256'] = hashlib.sha256(open(f'{args.out}/submission.csv', 'rb').read()).hexdigest()
    imp = pd.Series(np.mean([m.feature_importance('gain') for m in models], axis=0), index=feat_cols).sort_values(ascending=False)
    imp.to_csv(f'{args.out}/feature_importance.csv')
    json.dump(results, open(f'{args.out}/results.json', 'w'), indent=1, default=float)
    print('LOCKBOX macro-F1', round(results['lockbox']['macro_f1'], 4), 'submission dist', results['submission_label_dist'], f'{time.time()-t0:.0f}s')
