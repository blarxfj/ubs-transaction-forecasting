"""Deterministic, label-free temporal and recurring-stream feature extraction."""
import hashlib
import re
from pathlib import Path
import numpy as np
import pandas as pd

LABELS = ['cloud', 'gym', 'insurance', 'mobile', 'music', 'software', 'streaming', 'none']
CUTOFF = pd.Timestamp('2026-01-01', tz='UTC')
MCC = {5732: 0, 7997: 1, 6300: 2, 4814: 3, 5734: 5}
KEYWORDS = [r'cloud|backup|storage', r'gym|fitness|\bfit\b|urban|club|membership', r'insurance|policy|cover|safe', r'phone|contract|monthly plan|service bill', r'audio|member pass|\bpass\b', r'software|saas|productivity|\bprod\b|suite', r'video|media']
BACKGROUND = re.compile(r'salary|atm|fresh foods|pharmacy|hotel|booking|electronics|ride|coffee|grocery|neighborhood|marketplace|casual|dining|p2p|service fee')

def hashed(s):
    return int(hashlib.sha256(str(s).encode()).hexdigest(), 16)

def read_labels(data):
    train = pd.read_csv(Path(data)/'train_labels.csv').assign(source='train')
    valid = pd.read_csv(Path(data)/'valid_labels.csv')
    locked = valid.client_id.map(lambda s: hashed(s)%5 == 0)
    lock_ids = valid.loc[locked, 'client_id'].tolist()
    # Locked targets never enter a model, a diagnostic, or feature construction.
    valid = valid.loc[~locked].assign(source='valid')
    return pd.concat([train, valid], ignore_index=True), lock_ids

def transaction_features(df):
    df = df.copy()
    df['day'] = (pd.to_datetime(df.timestamp, utc=True)-CUTOFF).dt.total_seconds()/86400
    assert df.day.max() < 0, 'Future data must not enter features'
    df['mcc'] = pd.to_numeric(df.mcc)
    return df

def assign_families(g):
    """Combine MCC/text with same-client, similar-amount evidence."""
    n = len(g)
    evidence = np.zeros((n, 7))
    eligible = np.ones(n, bool)
    for i, row in enumerate(g.itertuples()):
        desc = str(row.description).lower()
        if BACKGROUND.search(desc) or row.type not in ('card_payment', 'refund'):
            eligible[i] = False
            continue
        if row.mcc in MCC:
            evidence[i, MCC[row.mcc]] += 1.0
        if row.mcc == 5812:
            evidence[i, [4, 6]] += .45
        for k, pat in enumerate(KEYWORDS):
            if re.search(pat, desc):
                evidence[i, k] += 1.4
        if 'service plan' in desc:
            evidence[i, 0] += 1.0
    amount = np.maximum(g.amount.to_numpy(), .1)
    distance = abs(np.log(amount[:, None]/amount[None, :]))
    kernel = np.exp(-.5*(distance/.035)**2)*eligible[:, None]*eligible[None, :]
    # Propagate evidence only within each client's own history; no label fitting.
    smooth = kernel@evidence
    score = evidence + 1.3*smooth / np.maximum(kernel.sum(axis=1, keepdims=True), 1)
    family = score.argmax(axis=1)
    family[~eligible] = -1
    confidence = score.max(axis=1)
    family[confidence == 0] = -1
    return family, confidence

def stats(t, a):
    t, a = np.asarray(t), np.asarray(a)
    out = {'n':len(t)}
    if not len(t):
        return out
    order = np.argsort(t); t=t[order]; a=a[order]
    age = -t[-1]
    out.update(age=age, first_age=-t[0], span=t[-1]-t[0], amount_mean=a.mean(), amount_last=a[-1], amount_med=np.median(a), amount_cv=a.std()/max(a.mean(),.01), amount_mad=np.median(abs(a-np.median(a)))/max(np.median(a),.01), amount_range=(a.max()-a.min())/max(a.mean(),.01))
    for w in [15,30,45,60,90,120,180,270]:
        out['n'+str(w)] = int((t>=-w).sum())
    for j in range(1,6):
        out['age'+str(j)] = -t[-j] if len(t)>=j else np.nan
    if len(t)>=2:
        gap=np.diff(t)
        out.update(gap_mean=gap.mean(),gap_med=np.median(gap),gap_std=gap.std(),gap_min=gap.min(),gap_max=gap.max(), gap_last=gap[-1], gap_recent=np.median(gap[-3:]), amount_delta=(a[-1]-a[0])/max(a.mean(),.01), amount_last_delta=(a[-1]-a[-2])/max(a.mean(),.01))
        out['amount_slope'] = np.polyfit(t-t.mean(),a/max(a.mean(),.01),1)[0]
        for v in ['gap_mean','gap_med','gap_recent']:
            out['due_'+v] = out[v]-age
            out['overdue_'+v] = age/max(out[v],1)
        for j in range(1,5):
            out['gap'+str(j)]=gap[-j] if len(gap)>=j else np.nan
        out['regular_amount'] = float(np.mean(abs(np.diff(a))/np.maximum(a[:-1],.01)<.035))
    # Calendar/cadence evidence, including skipped occurrences.
    for p in [7,14,30,60,90,365]:
        ang=t*2*np.pi/p
        z=np.mean(np.exp(1j*ang))
        out[f'phase_strength{p}']=abs(z)
        out[f'phase_due{p}']=(np.angle(z)/(2*np.pi)*p)%p
        out[f'last_due{p}']=p-age
        if len(t)>=2:
            gaps=np.diff(t); rounds=np.maximum(np.round(gaps/p),1)
            out[f'gap_error{p}']=np.mean(np.minimum(abs(gaps-rounds*p),p))/p
    return out

def one_client(g):
    g=g.sort_values('day').reset_index(drop=True)
    fam, conf=assign_families(g)
    g['family']=fam
    out={'total_n':len(g), 'history_age':-g.day.min(), 'all_age':-g.day.max(), 'n_refunds':int((g.type=='refund').sum()), 'n_currencies':g.currency.nunique()}
    per=[]
    for k in range(7):
        h=g[(g.family==k)&(g.type=='card_payment')]
        f=stats(h.day.to_numpy(),h.amount.to_numpy())
        r=g[(g.family==k)&(g.type=='refund')]
        f['refund_n']=len(r)
        f['refund_age']=-r.day.max() if len(r) else np.nan
        f['refund_n60']=int((r.day>=-60).sum())
        if len(h):
            # Densest relative-amount neighborhood isolates stable streams.
            a=h.amount.to_numpy(); t=h.day.to_numpy()
            d=abs(np.log(np.maximum(a[:,None],.1)/np.maximum(a[None,:],.1)))
            weights=(d<.04).sum(axis=1)
            core=d[np.argmax(weights)]<.04
            f.update({'core_'+key:v for key,v in stats(t[core],a[core]).items()})
            f['core_fraction']=core.mean()
            f['text_confidence']=float(conf[h.index].mean())
            # Split by amount when two independent subscriptions share a family.
            recent=np.flatnonzero(t>=-100)
            if len(recent):
                seed=recent[np.argmax(weights[recent])]
                sub=d[seed]<.07
                f.update({'recentcore_'+key:v for key,v in stats(t[sub],a[sub]).items()})
            for m in MCC:
                f['mcc_'+str(m)]=float((h.mcc==m).mean())
        per.append(f)
    out['family_n']=sum(f['n']>0 for f in per)
    out['active_family_n']=sum(f.get('age',999)<45 for f in per)
    return out,per

def build(data, cache, include_heldout=False):
    cache=Path(cache); cache.mkdir(parents=True,exist_ok=True)
    labels,lock_ids=read_labels(data)
    ids=set(labels.client_id)
    if include_heldout:
        ids.update(lock_ids)
        ids.update(pd.read_csv(Path(data)/'sample_submission.csv').client_id)
    rows={}
    for split in ['train','valid']+(['test'] if include_heldout else []):
        tx=transaction_features(pd.read_json(Path(data)/f'{split}_transactions.jsonl',lines=True))
        tx=tx[tx.client_id.isin(ids)]
        for cid,g in tx.groupby('client_id',sort=True):
            rows[cid]=one_client(g)
        print('features',split,len(rows),flush=True)
    features=sorted(set(key for _,per in rows.values() for f in per for key in f))
    global_keys=sorted(set(key for glob,_ in rows.values() for key in glob))
    client_ids=sorted(rows)
    tensors=np.full((len(rows),8,len(features)),np.nan)
    globals_=np.zeros((len(rows),len(global_keys)))
    for i,cid in enumerate(client_ids):
        glob,per=rows[cid]
        globals_[i]=[glob[k] for k in global_keys]
        for k in range(7):
            tensors[i,k]=[per[k].get(f,np.nan) for f in features]
    np.savez_compressed(cache/'features.npz',ids=client_ids,tensor=tensors,global_=globals_,features=features,global_keys=global_keys)
    return load(cache)

def load(cache):
    return dict(np.load(Path(cache)/'features.npz',allow_pickle=False))

def matrices(z):
    t=z['tensor']; g=z['global_']; names=list(z['features'])
    # Wide multiclass baseline.
    wide=np.concatenate([g,t[:,:7].reshape(len(t),-1)],axis=1)
    # Shared candidate representation: own stream plus summaries/relative ranks.
    valid=np.where(np.isnan(t[:,:7]),np.inf,t[:,:7])
    sorted_=np.sort(valid,axis=1)
    sorted_[~np.isfinite(sorted_)]=np.nan
    summary=sorted_[:,:3].reshape(len(t),-1)
    rank=np.sum(t[:,:,None,:]>t[:,None,:7,:],axis=2)
    delta=t-sorted_[:,0:1,:]
    candidate=np.concatenate([t,rank,delta,np.repeat(g[:,None,:],8,axis=1),np.repeat(summary[:,None,:],8,axis=1),np.tile(np.eye(8)[None,:,:],(len(t),1,1))],axis=2)
    return np.nan_to_num(wide,nan=-999,posinf=9999,neginf=-999),np.nan_to_num(candidate,nan=-999,posinf=9999,neginf=-999)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--cache',required=True);p.add_argument('--heldout',action='store_true');a=p.parse_args()
    build(a.data,a.cache,a.heldout)
