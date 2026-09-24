"""Train the per-family model on pseudo-cutoff data; emit scores for the real clients as a feature."""
import sys, os, numpy as np, pandas as pd, lightgbm as lgb, pickle, argparse, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from streams import *; from pairwise import to_long; from protocol import fold_of
ap = argparse.ArgumentParser(); ap.add_argument('--pseudo', default='pseudo.pkl'); ap.add_argument('--feats', default='feats2.pkl'); ap.add_argument('--out', default='pseudo_scores.pkl')
ap.add_argument('--rounds', type=int, default=0)
args = ap.parse_args(); t0=time.time()
Xp, yp = pickle.load(open(args.pseudo,'rb')); X, _ = pickle.load(open(args.feats,'rb'))
Lp = to_long(Xp); Lp['y'] = (Lp.client_id.map(yp) == Lp.fam).astype(float)
feat_cols = [c for c in Lp.columns if c not in ('client_id','fam','y')]
base = Lp.client_id.str.split('@').str[0]
hold = base.map(lambda c: fold_of(99, c) == 0).values   # 20% of base clients held out for early stopping
params = dict(objective='binary', learning_rate=0.05, num_leaves=31, min_data_in_leaf=100, feature_fraction=0.5, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=8, seed=0, deterministic=True, force_row_wise=True)
if args.rounds == 0:
    m = lgb.train(params, lgb.Dataset(Lp.loc[~hold, feat_cols], Lp.y[~hold]), num_boost_round=5000, valid_sets=[lgb.Dataset(Lp.loc[hold, feat_cols], Lp.y[hold])], callbacks=[lgb.early_stopping(100, verbose=False)])
    rounds = m.best_iteration; print('best rounds', rounds, 'holdout auc-ish logloss', m.best_score, f'{time.time()-t0:.0f}s', flush=True)
else: rounds = args.rounds
m = lgb.train(params, lgb.Dataset(Lp[feat_cols], Lp.y), num_boost_round=rounds)
L = to_long(X)
s = m.predict(L[feat_cols])
E = pd.DataFrame({'client_id': L.client_id.values, 'fam': L.fam.values, 's': s}).pivot(index='client_id', columns='fam', values='s')[FAM]
pickle.dump(E, open(args.out,'wb')); m.save_model(args.out.replace('.pkl','.txt'))
print('saved', E.shape, f'{time.time()-t0:.0f}s')
