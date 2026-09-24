"""Fixed-label, repeated hash-fold metrics and client bootstrap confidence intervals."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from features import LABELS,read_labels,hashed


def confusion_f1(y,pred,weight=None):
    cm=np.bincount(y*8+pred,weights=weight,minlength=64).reshape(8,8)
    den=cm.sum(0)+cm.sum(1)
    return np.divide(2*np.diag(cm),den,out=np.zeros(8),where=den>0)


def summarize(data,paths,bootstrap=2000):
    labels,_=read_labels(data)
    frames=[pd.read_csv(p).set_index('client_id') for p in paths]
    ids=frames[0].index
    labels=labels.set_index('client_id').loc[ids]
    y=labels.target_next_recurring_merchant.map(LABELS.index).to_numpy()
    valid=labels.source.to_numpy()=='valid'
    predictions=[]; folds=[]; results=[]
    for seed,frame in enumerate(frames):
        assert frame.index.equals(ids)
        fold=frame.fold.to_numpy()
        assert np.array_equal(fold,[hashed(f'{seed}:{cid}')%5 for cid in ids])
        pred=frame[LABELS].to_numpy().argmax(1)
        predictions.append(pred);folds.append(fold)
        per_fold=np.array([confusion_f1(y[fold==k],pred[fold==k]) for k in range(5)])
        valid_fold=np.array([confusion_f1(y[(fold==k)&valid],pred[(fold==k)&valid]) for k in range(5)])
        results.append({'seed':seed,'mean_fold_macro_f1':float(per_fold.mean()),'pooled_macro_f1':float(confusion_f1(y,pred).mean()),'fold_macro_f1':per_fold.mean(1).tolist(),'mean_fold_per_class_f1':dict(zip(LABELS,per_fold.mean(0).tolist())),'pooled_per_class_f1':dict(zip(LABELS,confusion_f1(y,pred).tolist())),'valid_mean_fold_macro_f1':float(valid_fold.mean()),'valid_pooled_macro_f1':float(confusion_f1(y[valid],pred[valid]).mean()),'train_pooled_macro_f1':float(confusion_f1(y[~valid],pred[~valid]).mean())})
    rng=np.random.default_rng(1729);samples=[];valid_samples=[]
    for _ in range(bootstrap):
        weight=np.bincount(rng.integers(len(y),size=len(y)),minlength=len(y))
        scores=[];vscores=[]
        for pred,fold in zip(predictions,folds):
            for k in range(5):
                m=fold==k
                scores.append(confusion_f1(y[m],pred[m],weight[m]).mean())
                m=m&valid
                vscores.append(confusion_f1(y[m],pred[m],weight[m]).mean())
        samples.append(np.mean(scores));valid_samples.append(np.mean(vscores))
    return {'n_clients':len(y),'n_train':int((~valid).sum()),'n_nonlock_valid':int(valid.sum()),'labels':LABELS,'primary_mean_macro_f1':float(np.mean([r['mean_fold_macro_f1'] for r in results])),'mean_pooled_macro_f1':float(np.mean([r['pooled_macro_f1'] for r in results])),'valid_mean_fold_macro_f1':float(np.mean([r['valid_mean_fold_macro_f1'] for r in results])),'valid_mean_pooled_macro_f1':float(np.mean([r['valid_pooled_macro_f1'] for r in results])),'mean_fold_per_class_f1':{label:float(np.mean([r['mean_fold_per_class_f1'][label] for r in results])) for label in LABELS},'bootstrap':{'unit':'client, same resample across repeated seeds','replicates':bootstrap,'seed':1729,'primary_percentile_95':np.quantile(samples,[.025,.975]).tolist(),'valid_percentile_95':np.quantile(valid_samples,[.025,.975]).tolist()},'seeds':results}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--oof-dir',required=True);p.add_argument('--output',required=True);p.add_argument('--bootstrap',type=int,default=2000);a=p.parse_args()
    result=summarize(a.data,[Path(a.oof_dir)/f'oof_seed{s}.csv' for s in range(3)],a.bootstrap)
    Path(a.output).write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
