import sys, numpy as np, pandas as pd, lightgbm as lgb, pickle, os, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from streams import *; from features import build_features; from protocol import *
ap = argparse.ArgumentParser(); ap.add_argument('--feats', default='feats2.pkl'); ap.add_argument('--rebuild', action='store_true'); ap.add_argument('--lr', type=float, default=0.03)
ap.add_argument('--leaves', type=int, default=15); ap.add_argument('--minleaf', type=int, default=20); ap.add_argument('--ff', type=float, default=0.5); ap.add_argument('--l2', type=float, default=1.0)
ap.add_argument('--drop', default=''); ap.add_argument('--seeds', default='0,1,2'); ap.add_argument('--save_oof', default='')
args = ap.parse_args()
lab = pd.concat([pd.read_csv('data/train_labels.csv'), pd.read_csv('data/valid_labels.csv')]).set_index('client_id').target_next_recurring_merchant
if os.path.exists(args.feats) and not args.rebuild:
    X, A = pickle.load(open(args.feats,'rb'))
else:
    tr = pd.concat([load('train'), load('valid'), load('test')])
    X, A = build_features(tr); pickle.dump((X,A), open(args.feats,'wb'))
if args.drop:
    X = X[[c for c in X.columns if not any(c.startswith(p) or c.endswith(p) for p in args.drop.split(','))]]
print(X.shape)
dev = [c for c in lab.index if not is_lockbox(c)]
Xd = X.loc[dev]; yd = lab.loc[dev].map(LABELS.index).values
valid_ids = set(pd.read_csv('data/valid_labels.csv').client_id)
params = dict(objective='multiclass', num_class=8, learning_rate=args.lr, num_leaves=args.leaves, min_data_in_leaf=args.minleaf, feature_fraction=args.ff, bagging_fraction=0.8, bagging_freq=1, lambda_l2=args.l2, verbose=-1, num_threads=8)
res=[]; resv=[]; oofs={}
for seed in [int(s) for s in args.seeds.split(',')]:
    folds = np.array([fold_of(seed, c) for c in dev]); oof = np.zeros((len(dev), 8)); its=[]
    for k in range(5):
        trn = folds!=k; tst = folds==k
        m = lgb.train({**params, 'seed': seed}, lgb.Dataset(Xd[trn], yd[trn]), num_boost_round=3000, valid_sets=[lgb.Dataset(Xd[tst], yd[tst])], callbacks=[lgb.early_stopping(100, verbose=False)])
        oof[tst] = m.predict(Xd[tst], num_iteration=m.best_iteration); its.append(m.best_iteration)
    pred = np.array(LABELS)[oof.argmax(1)]; ytrue = np.array(LABELS)[yd]
    f = macro_f1(ytrue, pred); isv = np.array([c in valid_ids for c in dev]); fv = macro_f1(ytrue[isv], pred[isv])
    print('seed', seed, 'iters', its, 'macroF1', round(f,4), 'valid-only', round(fv,4), {k: round(float(v),3) for k,v in per_class_f1(ytrue, pred).items()})
    res.append(f); resv.append(fv); oofs[seed]=oof
print('MEAN', round(np.mean(res),4), 'valid-only', round(np.mean(resv),4))
if args.save_oof: pickle.dump((dev, yd, oofs), open(args.save_oof,'wb'))
