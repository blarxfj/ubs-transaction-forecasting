"""UBS Transaction Activity Forecasting - reproducible baseline (rank + none-gate LightGBM).

Usage:
    python ubs_baseline.py --data DIR_WITH_UNZIPPED_FILES --out OUT_DIR [--jobs 8]

Steps: parse descriptions -> detect amount/currency streams -> per-(client, family) features ->
label-independent noise augmentation to test level -> LightGBM candidate ranker + none gate ->
reports (rules, leak demo, CV on valid folds at valid and test noise) -> test submission + contract check.
"""
import argparse, collections, os, random, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

# ----------------------------------------------------------------------------------------------
# 1. Vocabulary (recovered from the data: 73 tokens; 7 families x 4 base names, 4 generic names each)
# ----------------------------------------------------------------------------------------------
FAMS = ['cloud', 'gym', 'insurance', 'mobile', 'music', 'software', 'streaming']
CLASSES = FAMS + ['none']
FI = {f: i for i, f in enumerate(FAMS)}
FAM_NAMES = {
    'cloud': ['cloud access', 'cloud backup', 'service plan', 'storage plan'],
    'gym': ['urban gym', 'fitness monthly', 'fit club', 'gym membership'],
    'insurance': ['cover plan', 'policy premium', 'safe cover', 'insurance monthly', 'cover monthly'],
    'mobile': ['phone contract', 'service bill', 'monthly plan', 'digital plus'],
    'music': ['member pass', 'audio streaming', 'digital plus', 'premium plan'],
    'software': ['saas billing', 'software access', 'productivity suite', 'premium plan'],
    'streaming': ['media streaming', 'video access', 'digital plus', 'premium plan'],
}
FAM_MCC = {'mobile': '4814', 'cloud': '5732', 'software': '5734', 'insurance': '6300', 'gym': '7997',
           'streaming': '5812', 'music': '5812'}
GENERIC_SUB = ['subscription charge', 'monthly plan', 'member plan', 'digital service']
GENERIC_EVE = ['digital order', 'service payment', 'card purchase', 'merchant charge']
EVERYDAY = ['ride share', 'fresh foods', 'neighborhood market', 'grocery store', 'electronics shop',
            'online marketplace', 'coffee shop', 'casual dining', 'pharmacy', 'hotel booking']
OTHER = ['atm withdrawal', 'salary', 'p2p send', 'p2p receive', 'service fee']
ABBR_INV = {'dgtl': 'digital', 'prem': 'premium', 'mth': 'monthly', 'prod': 'productivity', 'stream': 'streaming'}
ABBR = {v: k for k, v in ABBR_INV.items()}
PREFIX = {'member', 'pay', 'billing'}
SUFFIX = {'core', 'online', 'service', 'plus', 'digital', 'dgtl'}
CARD_MCCS = ['4111', '4814', '5411', '5732', '5734', '5812', '5912', '6300', '7011', '7997']
ALL_BASES = set(sum(FAM_NAMES.values(), [])) | set(GENERIC_SUB) | set(GENERIC_EVE) | set(EVERYDAY) | set(OTHER)
TRUNC = collections.defaultdict(set)
for _b in ALL_BASES:
    _t = _b.split()
    if len(_t) > 1:
        TRUNC[_t[0]].add(_b); TRUNC[_t[-1]].add(_b)
BASE_FAMS = collections.defaultdict(set)
for _f, _ns in FAM_NAMES.items():
    for _n in _ns:
        BASE_FAMS[_n].add(_f)
_PARSE = {}
KIND_PRIORITY = {'fam': 4, 'generic_sub': 3, 'generic_eve': 2, 'everyday': 1, 'other': 0}


def parse(desc):
    """Strip up to 2 prefixes/suffixes, expand abbreviations, match a base name (exact or 1-token truncation).
    Returns (set of candidate base names, n stripped tokens, exact match?)."""
    if desc in _PARSE:
        return _PARSE[desc]
    toks = desc.split(); cands = []
    for npre in range(0, min(3, len(toks))):
        if npre and toks[npre - 1] not in PREFIX:
            break
        for nsuf in range(0, min(3, len(toks) - npre)):
            if nsuf and toks[len(toks) - nsuf] not in SUFFIX:
                break
            core = [ABBR_INV.get(t, t) for t in toks[npre:len(toks) - nsuf]]
            cs = ' '.join(core)
            if cs in ALL_BASES:
                cands.append((0, npre + nsuf, {cs}))
            elif len(core) == 1 and cs in TRUNC:
                cands.append((1, npre + nsuf, TRUNC[cs]))
    if cands:
        cands.sort(key=lambda c: (c[0], c[1]))
        res = (cands[0][2], cands[0][1], cands[0][0] == 0)
    else:
        bs = set()
        for t in toks:
            t = ABBR_INV.get(t, t)
            if t in TRUNC and t not in PREFIX | SUFFIX:
                bs |= TRUNC[t]
        res = (bs, 99, False)
    _PARSE[desc] = res
    return res


def evidence(desc):
    """-> (kind, {family: weight}); kind in fam | generic_sub | generic_eve | everyday | other | unknown."""
    bs, _, _ = parse(desc)
    if not bs:
        return 'unknown', {}
    kinds = collections.Counter(); fw = collections.Counter()
    for b in sorted(bs):  # sorted: set order depends on the per-process hash seed
        w = 1.0 / len(bs)
        if b in BASE_FAMS:
            kinds['fam'] += w
            for f in BASE_FAMS[b]:
                fw[f] += w / len(BASE_FAMS[b])
        if b in GENERIC_SUB: kinds['generic_sub'] += w
        if b in GENERIC_EVE: kinds['generic_eve'] += w
        if b in EVERYDAY: kinds['everyday'] += w
        if b in OTHER: kinds['other'] += w
    tot = sum(fw.values())
    kind = max(kinds, key=lambda k: (round(kinds[k], 9), KIND_PRIORITY[k]))  # deterministic tie-break
    return kind, ({f: v / tot for f, v in sorted(fw.items())} if tot else {})


# ----------------------------------------------------------------------------------------------
# 2. Loading
# ----------------------------------------------------------------------------------------------
def load(data, split):
    df = pd.read_json(os.path.join(data, f'{split}_transactions.jsonl'), lines=True,
                      dtype={'mcc': str, 'client_id': str})
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    df['date'] = df.timestamp.dt.tz_localize(None).dt.floor('D')
    return df


def load_labels(data, split):
    return pd.read_csv(os.path.join(data, f'{split}_labels.csv')).set_index('client_id').target_next_recurring_merchant


# ----------------------------------------------------------------------------------------------
# 3. Streams and features
# ----------------------------------------------------------------------------------------------
CUT = pd.Timestamp('2026-01-01')
SUB_KINDS = ('fam', 'generic_sub', 'unknown')
EPS_DESC, RHO_MCC, EPS_REF = 0.10, 0.30, 0.05
AMT = {}  # filled by amount_prior(): {'bins': array, 'logp': {fam: array}}


def row_evidence(df):
    ev = {d: evidence(d) for d in df.description.unique()}
    df = df.copy()
    df['kind'] = df.description.map(lambda d: ev[d][0])
    W = np.zeros((len(df), len(FAMS)))
    for j, d in enumerate(df.description.values):
        for f, w in ev[d][1].items():
            W[j, FI[f]] = w
    for f in FAMS:
        df['d_' + f] = W[:, FI[f]]
    df['day'] = (df.date - CUT).dt.days
    return df


def build_streams(g, tol=0.035):
    """Single-linkage clustering of log-amount within currency (streams are single-currency)."""
    out = []
    for _, gc in g.groupby('currency'):
        gc = gc.sort_values('amount')
        la = np.log(gc.amount.values)
        for ix in np.split(np.arange(len(gc)), np.where(np.diff(la) > tol)[0] + 1):
            out.append(gc.iloc[ix].sort_values('day'))
    return out


def amt_logp(a):
    bins = AMT['bins']
    b = int(np.clip(np.searchsorted(bins, np.log(a)) - 1, 0, len(bins) - 2))
    return np.array([AMT['logp'][f][b] for f in FAMS])


def desc_loglik(W, eps):
    inf = W.sum(axis=1) > 0
    if not inf.any():
        return np.zeros(len(FAMS))
    L = np.log((1 - eps) * W[inf] + eps / len(FAMS)).sum(axis=0)
    return L - L.max()


def mcc_loglik(mccs):
    L = np.zeros(len(FAMS))
    for m in mccs:
        L += np.log(np.array([(1 - RHO_MCC) * (m == FAM_MCC[f]) + RHO_MCC / 10 for f in FAMS]))
    return L - L.max()


def stream_record(s, refunds, main_cur):
    days = s.day.values; n = len(s); gaps = np.diff(days)
    per = float(np.median(gaps)) if n >= 2 else np.nan
    amt = float(np.median(s.amount))
    ref = refunds[(np.abs(refunds.amount / amt - 1) < 0.03) & (refunds.currency == s.currency.iloc[0])] if len(refunds) else refunds
    if len(ref):
        ref = ref[np.array([((d - days >= 0) & (d - days <= 10)).any() for d in ref.day.values], dtype=bool)]
    nref = len(ref)
    L = desc_loglik(s[['d_' + f for f in FAMS]].values, EPS_DESC) + mcc_loglik(s.mcc.values) + amt_logp(amt)
    if nref:  # refund rows are never noised in any split -> strong evidence
        L = L + desc_loglik(ref[['d_' + f for f in FAMS]].values, EPS_REF)
    post = np.exp(L - L.max()); post /= post.sum()
    last = int(days.max())
    return dict(n=n, first=int(days.min()), last=last, span=int(last - days.min()), per=per,
                gap_mad=float(np.median(np.abs(gaps - per))) if n >= 3 else np.nan, amt=amt,
                amt_cv=float(s.amount.std() / s.amount.mean()) if n >= 2 else 0.0,
                amt_trend=float(s.amount.iloc[-1] / s.amount.iloc[0]) if n >= 2 else 1.0, nref=nref,
                last_refunded=float(nref > 0 and ((ref.day.values - last >= 0) & (ref.day.values - last <= 10)).any()),
                last_ref=float(ref.day.max()) if nref else np.nan, odd_cur=float(s.currency.iloc[0] != main_cur),
                generic_frac=float((s.kind == 'generic_sub').mean()),
                mcc_top=float(s.mcc.value_counts(normalize=True).iloc[0]), post=post)


def _active(s):
    return s['last'] + (s['per'] if np.isfinite(s['per']) else 30) >= -7


def client_block(g, streams):
    t = g.type.value_counts()
    f = {'c_rows': len(g)}
    for typ in ['card_payment', 'topup', 'p2p_transfer', 'transfer', 'atm', 'refund', 'fee']:
        f['c_n_' + typ] = int(t.get(typ, 0))
    rf = g[g.type == 'refund']
    sub_rf = rf[rf.kind.isin(['fam', 'generic_sub'])]
    f['c_sub_refunds'] = len(sub_rf); f['c_eve_refunds'] = len(rf) - len(sub_rf)
    f['c_fee_pos'] = float((g.fee > 0).mean())
    main = g.currency.value_counts().index[0]
    f['c_oddcur'] = float((g.currency != main).mean())
    f['c_first_day'] = int(g.day.min())
    multi = [s for s in streams if s['n'] >= 2]
    act = [s for s in multi if _active(s)]
    ended = [s for s in multi if not _active(s)]
    f['c_n_streams'] = len(multi); f['c_n_active'] = len(act); f['c_n_ended'] = len(ended)
    f['c_n_active_ref'] = sum(s['nref'] > 0 for s in act); f['c_n_ended_ref'] = sum(s['nref'] > 0 for s in ended)
    nx = [s['last'] + s['per'] for s in act]
    f['c_min_next'] = min(nx) if nx else np.nan
    f['c_n_active_early'] = sum(-7 <= v <= 15 for v in nx)
    f['c_n_recent_singles'] = sum(1 for s in streams if s['n'] == 1 and s['last'] >= -45 and s['post'].max() > 0.5)
    f['c_max_act_n'] = max((s['n'] for s in act), default=0)
    f['c_min_act_first'] = min((s['first'] for s in act), default=0)
    f['c_frac_ref_streams'] = float(np.mean([s['nref'] > 0 for s in multi])) if multi else np.nan
    f['c_sub_ref_last90'] = int((sub_rf.day >= -90).sum())
    f['c_last_sub_ref'] = float(sub_rf.day.max()) if len(sub_rf) else np.nan
    return f


P_KEYS = ['p_prob', 'p_n', 'p_last', 'p_first', 'p_span', 'p_per', 'p_gap_mad', 'p_amt', 'p_amt_cv', 'p_amt_trend',
          'p_nref', 'p_ref_rate', 'p_last_refunded', 'p_last_ref', 'p_odd_cur', 'p_generic_frac', 'p_mcc_top',
          'p_next', 'p_overdue', 'p_active']


def family_block(f, streams):
    j = FI[f]; r = {}
    multi = [s for s in streams if s['n'] >= 2 and s['post'][j] >= 0.15]
    singles = [s for s in streams if s['n'] == 1 and s['post'][j] >= 0.15 and s['last'] >= -62]
    act = [s for s in multi if _active(s)]
    r['n_streams'] = float(sum(s['post'][j] for s in multi))
    r['n_active'] = float(sum(s['post'][j] for s in act))
    r['n_ended_ref'] = float(sum(s['post'][j] for s in multi if not _active(s) and s['nref'] > 0))
    pool = act if act else multi
    p = max(pool, key=lambda s: (s['post'][j] > 0.5, s['last'], s['n'])) if pool else None
    if p is not None:
        per = p['per'] if np.isfinite(p['per']) and p['per'] > 0 else 30.0
        vals = [p['post'][j], p['n'], p['last'], p['first'], p['span'], per, p['gap_mad'], p['amt'], p['amt_cv'],
                p['amt_trend'], p['nref'], p['nref'] / p['n'], p['last_refunded'], p['last_ref'], p['odd_cur'],
                p['generic_frac'], p['mcc_top'], p['last'] + per, -p['last'] / per, float(_active(p))]
        r.update(dict(zip(P_KEYS, vals)))
    else:
        r.update({k: np.nan for k in P_KEYS})
    s = max(singles, key=lambda s: (s['post'][j], s['last'])) if singles else None
    r['s_prob'] = s['post'][j] if s else np.nan
    r['s_last'] = s['last'] if s else np.nan
    r['s_amt_logp'] = amt_logp(s['amt'])[j] if s else np.nan
    r['s_n'] = len(singles)
    return r


# features whose value depends on the description/MCC noise level (they carry the train-only leak
# unless the training data is re-noised label-independently)
FEAT_LEAKY = ['p_prob', 's_prob', 'p_generic_frac', 'p_mcc_top']


def build_features(df):
    df = row_evidence(df)
    rows = []
    for cid, g in df.groupby('client_id', sort=True):
        main = g.currency.value_counts().index[0]
        cand = g[(g.type == 'card_payment') & g.kind.isin(SUB_KINDS)]
        refunds = g[g.type == 'refund']
        streams = [stream_record(s, refunds, main) for s in build_streams(cand)] if len(cand) else []
        cf = client_block(g, streams)
        fb = {f: family_block(f, streams) for f in FAMS}
        nexts = {f: (fb[f]['p_next'] if fb[f]['p_active'] == 1 else np.nan) for f in FAMS}
        vals = sorted(v for v in nexts.values() if np.isfinite(v))
        for f in FAMS:
            r = fb[f]; mine = nexts[f]
            others = [v for g2, v in nexts.items() if g2 != f and np.isfinite(v)]
            r['rel_next_minus_best_other'] = (mine - min(others)) if (others and np.isfinite(mine)) else np.nan
            r['n_other_active'] = len(others)
            r['next_rank'] = vals.index(mine) if np.isfinite(mine) else np.nan
            r['e_ref'] = float(refunds['d_' + f].sum())
            rows.append(dict(client_id=cid, family=f, family_id=FI[f], **r, **cf))
    return pd.DataFrame(rows)


def amount_prior(unl):
    """Per-family log-amount histogram from clean streams of the unlabeled set."""
    cp = unl[unl.type == 'card_payment']
    ev = {d: evidence(d) for d in cp.description.unique()}
    kind = cp.description.map(lambda d: ev[d][0])
    cp = cp[kind.isin(['fam', 'generic_sub'])]
    amts = collections.defaultdict(list)
    for _, g in cp.groupby('client_id'):
        g = g.sort_values('amount'); la = np.log(g.amount.values)
        for ix in np.split(np.arange(len(g)), np.where(np.diff(la) > 0.035)[0] + 1):
            s = g.iloc[ix]
            if len(s) < 3: continue
            fw = collections.Counter(next(iter(ev[d][1])) for d in s.description if len(ev[d][1]) == 1)
            if not fw: continue
            f, c = fw.most_common(1)[0]
            if c >= 2 and c >= 0.6 * len(s): amts[f].append(float(np.median(s.amount)))
    bins = np.linspace(np.log(1.5), np.log(800), 41)
    logp = {}
    for f in FAMS:
        h, _ = np.histogram(np.log(amts[f]), bins=bins); h = (h + 0.5) / (h + 0.5).sum(); logp[f] = np.log(h)
    return {'bins': bins, 'logp': logp}


# ----------------------------------------------------------------------------------------------
# 4. Noise augmentation (valid/test are much noisier than train; train has a label leak in its noise)
# ----------------------------------------------------------------------------------------------
TEST_LVL = dict(r_mask=0.50, r_mcc=0.28, r_affix=0.40, r_eve_mask=0.67, r_eve_mcc=0.12)
VALID_LVL = dict(r_mask=0.35, r_mcc=0.18, r_affix=0.40, r_eve_mask=0.47, r_eve_mcc=0.09)
VALID_TO_TEST = dict(p_eve_mask=0.38, p_eve_mcc=0.04, p_sub_mask=0.25, p_sub_mcc=0.10)


def _affix(desc, rng):
    toks = desc.split(); op = rng.random()
    if op < 0.3 and len(toks) >= 2:
        toks = [toks[0]] if rng.random() < 0.5 else [toks[-1]]
    elif op < 0.5:
        toks = [ABBR.get(t, t) if rng.random() < 0.7 else t for t in toks]
    if rng.random() < 0.35: toks = [rng.choice(sorted(PREFIX))] + toks
    if rng.random() < 0.6: toks = toks + [rng.choice(sorted(SUFFIX))]
    return ' '.join(toks)


def augment_redraw(df, r_mask, r_mcc, r_affix, r_eve_mask, r_eve_mcc, seed=0, min_post=0.6):
    """Label-independent re-draw of card-payment noise (removes train's masking<->none correlation)."""
    rng = random.Random(seed)
    d = row_evidence(df)
    fam_of = {}
    for _, g in d.groupby('client_id'):
        cand = g[(g.type == 'card_payment') & g.kind.isin(SUB_KINDS)]
        if not len(cand): continue
        main = g.currency.value_counts().index[0]; rf = g[g.type == 'refund']
        for s in build_streams(cand):
            if len(s) < 2: continue
            rec = stream_record(s, rf, main); j = int(np.argmax(rec['post']))
            if rec['post'][j] >= min_post:
                for i in s.index: fam_of[i] = FAMS[j]
    out = df.copy(); desc = out.description.to_dict(); mcc = out.mcc.to_dict(); kind = d.kind.to_dict()
    for i in out.index[out.type == 'card_payment']:
        k = kind[i]
        if i in fam_of:
            f = fam_of[i]
            if rng.random() < r_mask:
                desc[i] = rng.choice(GENERIC_SUB)
            else:
                base = desc[i] if k == 'fam' and f in evidence(desc[i])[1] else rng.choice(FAM_NAMES[f][:4])
                bs = parse(base)[0]; clean = next(iter(bs)) if len(bs) == 1 else base
                desc[i] = _affix(clean, rng) if rng.random() < r_affix else clean
            mcc[i] = rng.choice(CARD_MCCS) if rng.random() < r_mcc else FAM_MCC[f]
        elif k == 'everyday':
            if rng.random() < r_eve_mask: desc[i] = rng.choice(GENERIC_EVE)
            if rng.random() < r_eve_mcc: mcc[i] = rng.choice(CARD_MCCS)
    out['description'] = pd.Series(desc); out['mcc'] = pd.Series(mcc)
    return out


def augment_add(df, p_eve_mask, p_eve_mcc, p_sub_mask, p_sub_mcc, seed=0):
    """Extra noise on top of existing noise (valid -> test level)."""
    rng = random.Random(seed); df = df.copy()
    kind = df.description.map({d: evidence(d)[0] for d in df.description.unique()}).values
    desc = df.description.values.copy(); mcc = df.mcc.values.copy()
    for i in np.where((df.type == 'card_payment').values)[0]:
        if kind[i] == 'everyday':
            if rng.random() < p_eve_mask: desc[i] = rng.choice(GENERIC_EVE)
            if rng.random() < p_eve_mcc: mcc[i] = rng.choice(CARD_MCCS)
        elif kind[i] in ('fam', 'generic_sub'):
            if kind[i] == 'fam' and rng.random() < p_sub_mask: desc[i] = rng.choice(GENERIC_SUB)
            if rng.random() < p_sub_mcc: mcc[i] = rng.choice(CARD_MCCS)
    df['description'] = desc; df['mcc'] = mcc
    return df


# ----------------------------------------------------------------------------------------------
# 5. Model: candidate ranker (which family, trained on non-none clients) x none gate
# ----------------------------------------------------------------------------------------------
PARAMS = dict(objective='binary', learning_rate=0.03, num_leaves=15, min_child_samples=20, subsample=0.8,
              subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, n_estimators=400, verbose=-1,
              deterministic=True, force_col_wise=True, n_jobs=4)


def wide(long, drop=()):
    cols = [c for c in long.columns if not c.startswith('c_') and c not in ('client_id', 'family', 'family_id') and c not in drop]
    w = long.pivot(index='client_id', columns='family', values=cols)
    w.columns = [f'{f}__{c}' for c, f in w.columns]
    return w.join(long.groupby('client_id')[[c for c in long.columns if c.startswith('c_')]].first())


def fit_predict(train_tabs, eval_tabs, drop=(), seeds=(0, 1, 2)):
    """train_tabs: [(long, labels)], eval_tabs: [long]. Returns [DataFrame client x CLASSES]."""
    L = pd.concat([l.assign(_y=l.client_id.map(y)) for l, y in train_tabs], ignore_index=True)
    rcols = [c for c in L.columns if c not in ('client_id', 'family', '_y') and c not in drop]
    R = L[L._y != 'none']; rb = (R.family == R._y).astype(int)
    Ws = [wide(l, drop) for l, _ in train_tabs]
    W = pd.concat([w.reset_index(drop=True) for w in Ws], ignore_index=True)
    Wy = np.concatenate([(y.reindex(w.index) == 'none').astype(int).values for w, (_, y) in zip(Ws, train_tabs)])
    outs = [0] * len(eval_tabs)
    for sd in seeds:
        mr = lgb.LGBMClassifier(**PARAMS, random_state=sd).fit(R[rcols], rb, categorical_feature=['family_id'])
        mn = lgb.LGBMClassifier(**PARAMS, random_state=sd).fit(W, Wy)
        for k, le in enumerate(eval_tabs):
            sc = pd.Series(mr.predict_proba(le[rcols])[:, 1], index=le.index)
            P = le.assign(s=sc).pivot(index='client_id', columns='family', values='s')[FAMS]
            P = P.div(P.sum(axis=1), axis=0)
            X = wide(le, drop)[W.columns]
            pn = pd.Series(mn.predict_proba(X)[:, 1], index=X.index).reindex(P.index)
            P = P.mul(1 - pn, axis=0); P['none'] = pn
            outs[k] = outs[k] + P[CLASSES] / len(seeds)
    return outs


def decide(P):
    return pd.Series(np.array(CLASSES)[np.argmax(P.values, axis=1)], index=P.index)


def report(y, pred, title):
    y = y.reindex(pred.index)
    per = f1_score(y, pred, average=None, labels=CLASSES)
    mf = f1_score(y, pred, average='macro')
    print(f'{title:52s} macro-F1={mf:.4f} acc={(y == pred).mean():.4f} | ' +
          ' '.join(f'{c}={v:.3f}' for c, v in zip(CLASSES, per)), flush=True)
    return mf


# ----------------------------------------------------------------------------------------------
# 6. Main
# ----------------------------------------------------------------------------------------------
def _job(a):
    data, split, kind, kw, amt = a
    AMT.update(amt)
    df = load(data, split)
    if kind == 'redraw': df = augment_redraw(df, **kw)
    elif kind == 'add': df = augment_add(df, **kw)
    return build_features(df)


def rules(l, y, title):
    act = l[(l.p_active == 1) & (l.p_prob >= 0.4)]
    r1 = act.sort_values('p_next').groupby('client_id').family.first()
    sing = l[(l.s_prob >= 0.5) & (l.s_last >= -45)].sort_values('s_last', ascending=False).groupby('client_id').family.first()
    pred = r1.reindex(y.index).fillna(sing.reindex(y.index)).fillna('none')
    report(y, pred, f'{title} R1 earliest-next active stream else none')
    ref = l.groupby('client_id')[['c_n_active_ref', 'c_n_ended_ref']].first().sum(axis=1).reindex(y.index).fillna(0)
    pred = pred.copy(); pred[ref >= 2] = 'none'
    report(y, pred, f'{title} R4 R1 + none if >=2 refunded streams')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--jobs', type=int, default=8)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); t0 = time.time()
    AMT.update(amount_prior(load(a.data, 'unlabeled_pretrain')))
    ytr, yva = load_labels(a.data, 'train'), load_labels(a.data, 'valid')
    jobs = {'train': ('train', None, None), 'valid': ('valid', None, None), 'test': ('test', None, None),
            'trainT': ('train', 'redraw', dict(TEST_LVL, seed=0)), 'trainV': ('train', 'redraw', dict(VALID_LVL, seed=2)),
            'validT0': ('valid', 'add', dict(VALID_TO_TEST, seed=0)), 'validT1': ('valid', 'add', dict(VALID_TO_TEST, seed=1))}
    with ProcessPoolExecutor(a.jobs) as ex:
        T = dict(zip(jobs, ex.map(_job, [(a.data, s, k, kw, AMT) for s, k, kw in jobs.values()])))
    print(f'features built in {time.time() - t0:.0f}s', flush=True)

    print('\n== Rule baselines (no learning) ==')
    rules(T['valid'], yva, 'valid ')
    rules(T['validT0'], yva, 'validT')

    print('\n== LightGBM trained on train only, evaluated on valid ==')
    Pv, Pt = fit_predict([(T['train'], ytr)], [T['valid'], T['validT0']])
    report(yva, decide(Pv), 'train->valid  all features (train leak)')
    Pv, Pt = fit_predict([(T['train'], ytr)], [T['valid'], T['validT0']], drop=FEAT_LEAKY)
    report(yva, decide(Pv), 'train->valid  noise-level features dropped')
    report(yva, decide(Pt), 'train->validT noise-level features dropped')

    print('\n== 5-fold CV over valid clients; train = re-noised train + valid folds (+ test-noised copy) ==')
    ids = yva.index.values; oof_v, oof_t = [], []
    for fa, fb in StratifiedKFold(5, shuffle=True, random_state=0).split(ids, yva.values):
        ta, tb = set(ids[fa]), set(ids[fb])
        tr = [(T['trainT'], ytr), (T['trainV'], ytr), (T['valid'][T['valid'].client_id.isin(ta)], yva),
              (T['validT1'][T['validT1'].client_id.isin(ta)], yva)]
        pv, pt = fit_predict(tr, [T['valid'][T['valid'].client_id.isin(tb)], T['validT0'][T['validT0'].client_id.isin(tb)]])
        oof_v.append(pv); oof_t.append(pt)
    Pv, Pt = pd.concat(oof_v).reindex(yva.index), pd.concat(oof_t).reindex(yva.index)
    report(yva, decide(Pv), 'CV -> valid  (valid noise)')
    report(yva, decide(Pt), 'CV -> validT (test-level noise)')
    print(f"none-gate AUC valid={roc_auc_score(yva == 'none', Pv['none']):.4f} validT={roc_auc_score(yva == 'none', Pt['none']):.4f}")

    print('\n== Final model on all labelled data -> test submission ==')
    final = [(T['trainT'], ytr), (T['trainV'], ytr), (T['valid'], yva), (T['validT1'], yva)]
    (Ptest,) = fit_predict(final, [T['test']])
    sub_ref = pd.read_csv(os.path.join(a.data, 'sample_submission.csv'))
    pred = decide(Ptest).reindex(sub_ref.client_id)
    sub = pd.DataFrame({'client_id': sub_ref.client_id.values, 'predicted_next_recurring_merchant': pred.values})
    # contract checks
    assert list(sub.columns) == list(sub_ref.columns)
    assert len(sub) == len(sub_ref) and set(sub.client_id) == set(sub_ref.client_id) and not sub.client_id.duplicated().any()
    assert sub.predicted_next_recurring_merchant.isin(CLASSES).all()
    path = os.path.join(a.out, 'submission.csv'); sub.to_csv(path, index=False)
    Ptest.to_csv(os.path.join(a.out, 'test_probabilities.csv'))
    print('submission written:', path, 'rows', len(sub))
    print('predicted class distribution:', sub.predicted_next_recurring_merchant.value_counts().to_dict())
    print(f'total {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
