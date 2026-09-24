"""Transparent recurrence heuristic baselines on an already built feature cache."""
import argparse,json
from pathlib import Path
import numpy as np
from features import LABELS,load,read_labels,hashed
from evaluate import confusion_f1

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--cache',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    z=load(a.cache);lab,_=read_labels(a.data);lab=lab.set_index('client_id').loc[z['ids']];y=lab.target_next_recurring_merchant.map(LABELS.index).values
    folds=np.array([hashed(f'0:{cid}')%5 for cid in z['ids']]);valid=lab.source.values=='valid';t=z['tensor'][:,:7];names=list(z['features']);get=lambda key:t[:,:,names.index(key)]
    results=[]
    def record(name,pred):
        results.append({'name':name,'seed0_mean_fold_macro_f1':float(np.mean([confusion_f1(y[folds==k],pred[folds==k]).mean() for k in range(5)])),'pooled_macro_f1':float(confusion_f1(y,pred).mean()),'valid_pooled_macro_f1':float(confusion_f1(y[valid],pred[valid]).mean())})
    record('always none',np.full(len(y),7))
    for prefix in ['','core_','recentcore_']:
        for threshold in [35,45,60,90]:
            age=get(prefix+'age');n=get(prefix+'n');gap=get(prefix+'gap_med');due=gap-age
            eligible=(n>=2)&(age<=threshold)&(due>-7)
            v=np.where(eligible,abs(due),999);pred=v.argmin(1);pred[v.min(1)==999]=7
            record(f'{prefix or "all_"}earliest due, maximum age {threshold}',pred)
    Path(a.output).write_text(json.dumps(results,indent=2));print(json.dumps(results,indent=2))

if __name__=='__main__':main()
