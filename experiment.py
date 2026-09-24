"""Hash-fold cross-validation with no lockbox access."""
import argparse,json,time
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRanker, Pool
from sklearn.metrics import f1_score,classification_report
from features import LABELS,hashed,read_labels,load,matrices


def score(y,p):
    return float(f1_score(y,p.argmax(axis=1),labels=list(range(8)),average='macro'))


def fit_predict(x,y,tr,va,mode,iterations,depth,lr,seed,weights=False):
    params=dict(iterations=iterations,depth=depth,learning_rate=lr,l2_leaf_reg=5,random_seed=seed,thread_count=4,verbose=False,allow_writing_files=False)
    if mode=='wide':
        model=CatBoostClassifier(**params,loss_function='MultiClass',auto_class_weights='SqrtBalanced' if weights else None)
        model.fit(x[tr],y[tr]); p=model.predict_proba(x[va])
    elif mode=='candidate':
        model=CatBoostClassifier(**params,loss_function='Logloss')
        target=(np.arange(8)[None,:]==y[:,None]).reshape(-1).astype(int)
        model.fit(x[tr].reshape(-1,x.shape[-1]), target.reshape(-1,8)[tr].reshape(-1))
        p=model.predict_proba(x[va].reshape(-1,x.shape[-1]))[:,1].reshape(-1,8)
        p/=p.sum(axis=1,keepdims=True)
    elif mode.startswith('lgbm'):
        from lightgbm import LGBMRegressor, LGBMClassifier
        def objective(y_true,raw):
            raw=raw.reshape(-1,8);prob=np.exp(raw-raw.max(1,keepdims=True));prob/=prob.sum(1,keepdims=True)
            return (prob-y_true.reshape(-1,8)).reshape(-1),(2*prob*(1-prob)+1e-6).reshape(-1)
        kw=dict(n_estimators=iterations,max_depth=depth,num_leaves=2**depth,learning_rate=lr,min_child_samples=40,reg_lambda=5,colsample_bytree=.9,max_bin=63,n_jobs=4,random_state=seed,deterministic=True,force_col_wise=True,verbosity=-1)
        target=(np.arange(8)[None,:]==y[:,None]).astype(int)
        if mode=='lgbm_rank':
            model=LGBMRegressor(**kw,objective=objective)
        else:
            model=LGBMClassifier(**kw)
        model.fit(x[tr].reshape(-1,x.shape[-1]),target[tr].reshape(-1))
        if mode=='lgbm_rank':
            raw=model.predict(x[va].reshape(-1,x.shape[-1])).reshape(-1,8)
            p=np.exp(raw-raw.max(1,keepdims=True));p/=p.sum(1,keepdims=True)
        else:
            p=model.predict_proba(x[va].reshape(-1,x.shape[-1]))[:,1].reshape(-1,8);p/=p.sum(1,keepdims=True)
    else:
        model=CatBoostRanker(**params,loss_function='QuerySoftMax')
        target=(np.arange(8)[None,:]==y[:,None]).astype(int)
        model.fit(Pool(x[tr].reshape(-1,x.shape[-1]),target[tr].reshape(-1),group_id=np.repeat(np.arange(len(tr)),8)))
        p=model.predict(x[va].reshape(-1,x.shape[-1])).reshape(-1,8)
        p=np.exp(p-p.max(axis=1,keepdims=True));p/=p.sum(axis=1,keepdims=True)
    return p,model


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',required=True);ap.add_argument('--cache',required=True);ap.add_argument('--output',required=True);ap.add_argument('--mode',choices=['wide','candidate','ranker','lgbm_rank','lgbm'],default='candidate');ap.add_argument('--iterations',type=int,default=800);ap.add_argument('--depth',type=int,default=5);ap.add_argument('--lr',type=float,default=.04);ap.add_argument('--seeds',default='0');ap.add_argument('--weights',action='store_true');ap.add_argument('--compact',action='store_true');ap.add_argument('--folds',default='0,1,2,3,4');a=ap.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    z=load(a.cache);labels,_=read_labels(a.data);labels=labels.set_index('client_id').loc[z['ids']]
    y=np.array([LABELS.index(v) for v in labels.target_next_recurring_merchant]);valid=labels.source.to_numpy()=='valid'
    wide,candidates=matrices(z,a.compact);x=wide if a.mode=='wide' else candidates
    result={'configuration':vars(a),'n_clients':len(y),'features':x.shape[-1],'seeds':{}}
    for seed in map(int,a.seeds.split(',')):
        folds=np.array([hashed(f'{seed}:{cid}')%5 for cid in z['ids']]);p=np.zeros((len(y),8)); fs=[]; start=time.time()
        for fold in map(int,a.folds.split(',')):
            tr=np.flatnonzero(folds!=fold);va=np.flatnonzero(folds==fold)
            p[va],model=fit_predict(x,y,tr,va,a.mode,a.iterations,a.depth,a.lr,seed,a.weights)
            fs.append(score(y[va],p[va])); print('fold',seed,fold,'macro',fs[-1],'seconds',round(time.time()-start,1),flush=True)
            np.savez_compressed(out/f'partial_seed{seed}.npz',p=p,folds=folds)
        if a.folds!='0,1,2,3,4':
            (out/'partial_metrics.json').write_text(json.dumps({'configuration':vars(a),'fold_macro_f1':fs},indent=2))
            continue
        result['seeds'][str(seed)]={'pooled_macro_f1':score(y,p),'mean_fold_macro_f1':float(np.mean(fs)),'fold_macro_f1':fs,'valid_macro_f1':score(y[valid],p[valid]),'per_class_f1':dict(zip(LABELS,f1_score(y,p.argmax(axis=1),labels=range(8),average=None).tolist()))}
        frame=pd.DataFrame(p,columns=LABELS);frame.insert(0,'fold',folds);frame.insert(0,'client_id',z['ids']);frame.to_csv(out/f'oof_seed{seed}.csv',index=False,float_format='%.12g')
        print(json.dumps(result['seeds'][str(seed)]),flush=True)
        (out/'metrics.json').write_text(json.dumps(result,indent=2))
    if not result['seeds']:
        return
    result['mean_macro_f1']=float(np.mean([v['mean_fold_macro_f1'] for v in result['seeds'].values()]))
    result['mean_pooled_macro_f1']=float(np.mean([v['pooled_macro_f1'] for v in result['seeds'].values()]))
    (out/'metrics.json').write_text(json.dumps(result,indent=2))

if __name__=='__main__':main()
