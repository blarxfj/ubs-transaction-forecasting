import pandas as pd, numpy as np
FAM = ['cloud','gym','insurance','mobile','music','software','streaming']
LABELS = FAM + ['none']
MCC2FAM = {4814:'mobile',5732:'cloud',5734:'software',6300:'insurance',7997:'gym'}
KW = {
 'cloud': ['cloud','backup','storage','service plan'],
 'gym': ['gym','fit','fitness','club','urban'],
 'insurance': ['cover','insurance','policy','safe'],
 'mobile': ['phone','contract','service bill','bill'],
 'music': ['audio','member pass','pass'],
 'software': ['saas','productivity','suite','software','prod'],
 'streaming': ['media','video','stream','streaming'],
}
NOISE_DESC = {'merchant charge','service payment','card purchase','digital order'}
NONREC = {'coffee shop','casual dining','electronics shop','online marketplace','fresh foods','grocery store','neighborhood market','pharmacy','hotel booking','ride share','atm withdrawal','salary','p2p send','p2p receive','service fee'}
def kw_family(desc):
    toks = desc.split()
    for fam, kws in KW.items():
        for k in kws:
            if ' ' in k:
                if k in desc: return fam
            elif k in toks: return fam
    return None
def strip_noise(desc):
    toks = [t for t in desc.split() if t not in ('pay','billing','member','core','digital','online','plus','service')]
    return ' '.join(toks)
def load(n):
    d = pd.read_json(f'data/{n}_transactions.jsonl', lines=True)
    d['ts'] = pd.to_datetime(d.timestamp, utc=True).dt.tz_localize(None)
    d['day'] = (d.ts - pd.Timestamp('2026-01-01')).dt.total_seconds()/86400.0
    d['kwfam'] = d.description.map(kw_family)
    d['mccfam'] = d.mcc.map(MCC2FAM)
    d['isnoise'] = d.description.isin(NOISE_DESC)
    d['isnonrec'] = d.description.isin(NONREC)
    return d

def extract_streams(tr, tol=0.03):
    """Return DataFrame of streams: one row per (client, amount-cluster)."""
    c = tr[(tr.direction=='out') & (tr.type=='card_payment') & (~tr.isnoise) & (~tr.isnonrec)].copy()
    refunds = tr[(tr.type=='refund') & (~tr.isnoise) & (~tr.isnonrec)]
    rows=[]
    ref_by_client = {cid: g for cid, g in refunds.groupby('client_id')}
    for cid, g in c.groupby('client_id'):
        g = g.sort_values('amount'); la = np.log(g.amount.values)
        cl = np.concatenate([[0], np.cumsum(np.diff(la) > tol)])
        g = g.assign(cl=cl)
        rf = ref_by_client.get(cid)
        for k, gg in g.groupby('cl'):
            named = gg.kwfam.dropna()
            if len(named):
                vc = named.value_counts(); fam = vc.index[0]; fam_conf = vc.iloc[0]/len(gg); n_named=len(named)
            else:
                mv = gg.mccfam.dropna()
                if len(mv): fam = mv.value_counts().index[0]; fam_conf = 0.3; n_named=0
                elif (gg.mcc==5812).any(): fam='music_or_streaming'; fam_conf=0.3; n_named=0
                else: fam='unknown'; fam_conf=0.0; n_named=0
            d = np.sort(gg.day.values); gaps = np.diff(d)
            amt = gg.amount.median()
            nref = 0; last_ref = np.nan
            if rf is not None:
                m = rf[(np.abs(np.log(rf.amount/amt)) < 0.1)]
                nref = len(m); last_ref = m.day.max() if len(m) else np.nan
            rows.append(dict(client_id=cid, cl=k, fam=fam, fam_conf=fam_conf, n=len(d), n_named=n_named, first=d[0], last=d[-1],
                             medgap=np.median(gaps) if len(gaps) else np.nan, gap_std=np.std(gaps) if len(gaps)>1 else np.nan,
                             min_gap=gaps.min() if len(gaps) else np.nan, max_gap=gaps.max() if len(gaps) else np.nan,
                             amt=amt, amt_cv=gg.amount.std()/amt if len(gg)>1 else 0.0, nref=nref, last_ref=last_ref,
                             mcc_main=gg.mcc.value_counts().index[0], fee_sum=gg.fee.sum(), n_generic=int(gg.kwfam.isna().sum())))
    return pd.DataFrame(rows)
