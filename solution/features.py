"""Feature construction: family-first event assignment, then per-family stream statistics."""
import numpy as np, pandas as pd
from streams import FAM, MCC2FAM

FAM_MAIN_MCC = {'cloud':5732,'gym':7997,'insurance':6300,'mobile':4814,'music':5812,'software':5734,'streaming':5812}
GENERIC = {'monthly plan','member plan','digital service','subscription charge'}
CLEAN = {'cloud access','cloud backup','service plan','storage plan','urban gym','gym membership','fit club','fitness monthly',
         'cover plan','safe cover','policy premium','insurance monthly','monthly plan','phone contract','service bill','digital plus',
         'member pass','audio streaming','premium plan','saas billing','productivity suite','software access','media streaming','video access'} | GENERIC

def assign_family(g, link_tol=0.08):
    """g: candidate recurring events of one client (sorted by day). Returns family per event."""
    fam = g.kwfam.copy()
    named = g[g.kwfam.notna()]
    if len(named):
        # amount-link generic/unnamed events to the nearest named event amount (per family median amounts)
        # build family amount anchors: cluster named amounts per family
        anchors = []  # (fam, amt)
        for f, gf in named.groupby('kwfam'):
            a = np.sort(np.log(gf.amount.values))
            cl = np.concatenate([[0], np.cumsum(np.diff(a) > 0.04)])
            for k in np.unique(cl):
                if (cl==k).sum() >= 1: anchors.append((f, np.median(a[cl==k]), (cl==k).sum()))
        A = np.array([x[1] for x in anchors]); W = np.array([x[2] for x in anchors])
        for i in np.where(fam.isna())[0]:
            la = np.log(g.amount.iloc[i]); dist = np.abs(A - la)
            # prefer anchors with more support, within tolerance
            ok = dist < link_tol
            if ok.any():
                j = np.argmin(dist / np.sqrt(W))  # weight by support
                if dist[j] < link_tol: fam.iloc[i] = anchors[j][0]; continue
                j = np.argmin(np.where(ok, dist, 9)); fam.iloc[i] = anchors[j][0]; continue
    # remaining: MCC
    rem = fam.isna()
    fam[rem] = g.mccfam[rem]
    rem = fam.isna()
    fam[rem & (g.mcc==5812)] = 'amb5812'
    fam = fam.fillna('unknown')
    return fam

def timeline_feats(g, prefix, allref):
    """g: events of one family for one client, sorted by day."""
    f = {}
    d = g.day.values; a = g.amount.values; n = len(d)
    f[prefix+'_n'] = n
    if n == 0: return f
    gaps = np.diff(d)
    f[prefix+'_first'] = d[0]; f[prefix+'_last'] = d[-1]; f[prefix+'_span'] = d[-1]-d[0]
    f[prefix+'_n_named'] = g.kwfam.notna().sum(); f[prefix+'_frac_generic'] = g.kwfam.isna().mean()
    f[prefix+'_generic_last3'] = g.kwfam.tail(3).isna().sum(); f[prefix+'_generic_last1'] = float(g.kwfam.iloc[-1] is None or pd.isna(g.kwfam.iloc[-1]))
    noisy = ~g.description.isin(CLEAN)
    f[prefix+'_frac_noisy'] = noisy.mean(); f[prefix+'_noisy_last3'] = noisy.tail(3).sum(); f[prefix+'_n_noisy'] = noisy.sum()
    isgen = g.description.isin(GENERIC); f[prefix+'_frac_gen_exact'] = isgen.mean()
    main = FAM_MAIN_MCC.get(prefix)
    if main is not None:
        sw = (g.mcc.values != main)
        f[prefix+'_swap_frac'] = sw.mean(); f[prefix+'_swap_last1'] = float(sw[-1]); f[prefix+'_swap_last3'] = sw[-3:].sum(); f[prefix+'_n_swap'] = sw.sum()
        f[prefix+'_noise_score'] = sw.mean() + noisy.mean() + isgen.mean() + (g.fee.values > 0).mean()
        f[prefix+'_n_clean'] = ((~sw) & (~noisy.values) & (~isgen.values)).sum()
    f[prefix+'_n_mcc'] = g.mcc.nunique(); f[prefix+'_n_desc'] = g.description.nunique()
    f[prefix+'_fee_frac'] = (g.fee>0).mean(); f[prefix+'_fee_last'] = float(g.fee.iloc[-1]>0)
    f[prefix+'_n_cur'] = g.currency.nunique(); f[prefix+'_cur_last_diff'] = float(g.currency.iloc[-1] != g.currency.mode().iloc[0])
    la = np.log(a); med = np.median(la)
    f[prefix+'_logamt'] = med; f[prefix+'_amt_cv'] = np.std(la); f[prefix+'_amt_range'] = la.max()-la.min()
    f[prefix+'_last_amt_dev'] = la[-1]-med; f[prefix+'_last_amt_absdev'] = abs(la[-1]-med)
    f[prefix+'_max_absdev'] = np.max(np.abs(la-med)); f[prefix+'_n_outlier'] = (np.abs(la-med)>0.03).sum()
    if n >= 3: f[prefix+'_amt_slope'] = np.polyfit(d, la, 1)[0]*30
    for w in [30, 45, 60, 90, 180]:
        f[prefix+f'_n{w}'] = (d > -w).sum()
    if n >= 2:
        mg = np.median(gaps); f[prefix+'_medgap'] = mg; f[prefix+'_meangap'] = gaps.mean(); f[prefix+'_gap_std'] = gaps.std()
        f[prefix+'_min_gap'] = gaps.min(); f[prefix+'_max_gap'] = gaps.max(); f[prefix+'_last_gap'] = gaps[-1]; f[prefix+'_last_gap_ratio'] = gaps[-1]/mg if mg>0 else np.nan
        f[prefix+'_due'] = d[-1] + mg; f[prefix+'_overdue'] = (-d[-1]) / mg if mg>0 else np.nan
        f[prefix+'_due_mean'] = d[-1] + gaps.mean()
        f[prefix+'_n_dup'] = (gaps < 5).sum(); f[prefix+'_n_missed'] = (gaps > 1.6*mg).sum()
        # regularity: fraction of gaps within 20% of median
        f[prefix+'_reg'] = (np.abs(gaps/mg - 1) < 0.2).mean()
        # expected count in last 90 days vs observed
        f[prefix+'_cov90'] = (d > -90).sum() * mg / 90.0
        # day-of-month consistency
        dom = g.ts.dt.day.values; f[prefix+'_dom_std'] = np.std(dom)
    # refunds: matched by amount (within 10%) or by keyword of family
    if allref is not None and len(allref):
        rm = allref[(np.abs(np.log(allref.amount.values/np.exp(med))) < 0.1) | (allref.kwfam.values == prefix)]
        f[prefix+'_nref'] = len(rm); f[prefix+'_ref_last'] = rm.day.max() if len(rm) else -999
        f[prefix+'_ref_after_last'] = float((rm.day > d[-1]).any()) if len(rm) else 0.0
        f[prefix+'_nref60'] = (rm.day > -60).sum(); f[prefix+'_nref_frac'] = len(rm)/n
    else:
        f[prefix+'_nref'] = 0; f[prefix+'_ref_last'] = -999; f[prefix+'_ref_after_last'] = 0.0; f[prefix+'_nref60'] = 0; f[prefix+'_nref_frac'] = 0.0
    return f

def client_feats(g):
    """g: all transactions of one client sorted by day."""
    f = {}
    f['n_txn'] = len(g); f['first_txn'] = g.day.min(); f['last_txn'] = g.day.max()
    for t in ['card_payment','topup','p2p_transfer','transfer','atm','refund','fee']:
        f['n_'+t] = (g.type==t).sum()
    f['n_in'] = (g.direction=='in').sum(); f['n_cur'] = g.currency.nunique()
    f['main_cur'] = ['chf','eur','usd','gbp'].index(g.currency.value_counts().index[0])
    f['n_noise'] = g.isnoise.sum(); f['n_nonrec'] = g.isnonrec.sum(); f['fee_sum'] = g.fee.sum(); f['n_feepos'] = (g.fee>0).sum()
    f['salary_sum'] = g[g.description=='salary'].amount.sum(); f['out_sum'] = g[g.direction=='out'].amount.sum()
    rf = g[g.type=='refund']
    for w in [30,60,90,180]:
        f[f'n_ref{w}'] = (rf.day > -w).sum(); f[f'n_txn{w}'] = (g.day > -w).sum()
    f['ref_rec_named'] = rf.kwfam.notna().sum(); f['ref_generic'] = (rf.description.isin(GENERIC)).sum()
    f['ref_nonrec'] = rf.isnonrec.sum()
    return f

def build_features(tr):
    tr = tr.sort_values(['client_id','day']).reset_index(drop=True)
    tr = tr.assign(cand=(tr.direction=='out') & (tr.type=='card_payment') & (~tr.isnoise) & (~tr.isnonrec))
    rows = {}; assign = []
    for cid, g in tr.groupby('client_id', sort=False):
        f = client_feats(g)
        c = g[g.cand]
        rf = g[g.type=='refund']
        fam = assign_family(c) if len(c) else pd.Series(dtype=object)
        c = c.assign(fam=fam.values)
        assign.append(c[['client_id','day','amount','description','mcc','fam']])
        f['n_cand'] = len(c); f['n_amb'] = (c.fam=='amb5812').sum(); f['n_unknown'] = (c.fam=='unknown').sum()
        f['n_fam_present'] = c.fam.isin(FAM).sum() and c[c.fam.isin(FAM)].fam.nunique()
        for fm in FAM + ['amb5812','unknown']:
            f.update(timeline_feats(c[c.fam==fm], fm, rf))
        # cross-family: ranks of due among families with n>=2 and last > -75
        act = []
        for fm in FAM:
            if f.get(fm+'_n',0) >= 2 and f[fm+'_last'] > -75 and not np.isnan(f.get(fm+'_due', np.nan)):
                act.append((fm, f[fm+'_due'], f[fm+'_n'], f[fm+'_last']))
        f['n_active'] = len(act)
        if act:
            dues = sorted(act, key=lambda x: x[1]); ns = sorted(act, key=lambda x: -x[2]); lasts = sorted(act, key=lambda x: -x[3])
            f['min_due'] = dues[0][1]; f['max_n_active'] = ns[0][2]; f['max_last_active'] = lasts[0][3]
            for fm in FAM:
                f[fm+'_due_rank'] = next((i for i,x in enumerate(dues) if x[0]==fm), -1)
                f[fm+'_n_rank'] = next((i for i,x in enumerate(ns) if x[0]==fm), -1)
                f[fm+'_last_rank'] = next((i for i,x in enumerate(lasts) if x[0]==fm), -1)
                if f.get(fm+'_n',0)>=2 and fm+'_due' in f:
                    f[fm+'_due_minus_min'] = f[fm+'_due'] - dues[0][1]
        rows[cid] = f
    X = pd.DataFrame.from_dict(rows, orient='index')
    return X, pd.concat(assign, ignore_index=True)
