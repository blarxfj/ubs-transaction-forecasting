"""Reproduce development evaluation or fit the frozen recipe and predict."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from features import LABELS,hashed,read_labels,load,build,matrices,matrix_feature_names
from experiment import fit_predict
from evaluate import summarize,confusion_f1


def recipe(path):
    config=json.loads(Path(path).read_text())
    assert config['none_weight']>0
    assert abs(sum(c['weight'] for c in config['components'])-1)<1e-9
    return config


def adjust(p,config):
    p=p.copy();p[:,7]*=config['none_weight'];p/=p.sum(1,keepdims=True)
    return p


def write_prob(path,ids,p,fold=None):
    assert p.shape==(len(ids),8)
    assert np.all(np.isfinite(p)) and np.all(p>=0)
    assert np.allclose(p.sum(1),1)
    frame=pd.DataFrame(p,columns=LABELS)
    if fold is not None:frame.insert(0,'fold',fold)
    frame.insert(0,'client_id',ids)
    frame.to_csv(path,index=False,float_format='%.12g')


def cv(data,work,out,config,seeds):
    cache=work/'development_features'
    z=load(cache) if (cache/'features.npz').exists() else build(data,cache)
    lab,_=read_labels(data);lab=lab.set_index('client_id').loc[z['ids']]
    y=lab.target_next_recurring_merchant.map(LABELS.index).to_numpy()
    for seed in seeds:
        folds=np.array([hashed(f'{seed}:{c}')%5 for c in z['ids']]);p=np.zeros((len(y),8))
        for ci,c in enumerate(config['components']):
            wide,candidates=matrices(z,c.get('compact',False));x=wide if c['mode']=='wide' else candidates
            component=np.zeros_like(p)
            for fold in range(5):
                tr=np.flatnonzero(folds!=fold);va=np.flatnonzero(folds==fold)
                component[va],_=fit_predict(x,y,tr,va,c['mode'],c['iterations'],c['depth'],c['lr'],seed)
                print('cv',seed,ci,fold,float(confusion_f1(y[va],component[va].argmax(1)).mean()),flush=True)
            write_prob(out/f'component{ci}_seed{seed}.csv',z['ids'],component,folds)
            p+=c['weight']*component
        p=adjust(p,config)
        write_prob(out/f'oof_seed{seed}.csv',z['ids'],p,folds)
        if seed==0:write_prob(out/'oof.csv',z['ids'],p,folds)
    if all((out/f'oof_seed{s}.csv').exists() for s in range(3)):
        result=summarize(data,[out/f'oof_seed{s}.csv' for s in range(3)])
        (out/'metrics.json').write_text(json.dumps(result,indent=2))
        print('primary mean macro-F1',result['primary_mean_macro_f1'],flush=True)


def final(data,work,out,config,config_path,evaluate_lockbox,test_only=False):
    # This stage is deliberately separate from all model selection.
    if evaluate_lockbox and test_only:
        raise ValueError('Test-only determinism checks must not evaluate the lockbox.')
    manifest={'recipe_sha256':hashlib.sha256(Path(config_path).read_bytes()).hexdigest(),'config':config,'training_clients':'train plus non-lockbox validation','seed':0}
    if evaluate_lockbox and (out/'lockbox_metrics.json').exists():
        raise RuntimeError('Lockbox already evaluated here. Reuse the recorded result, not another selection round.')
    source=Path(__file__).parent
    manifest['source_sha256']={name:hashlib.sha256((source/name).read_bytes()).hexdigest() for name in ['features.py','experiment.py','evaluate.py','run.py','requirements.txt']}
    if (out/'frozen_manifest.json').exists():
        previous=json.loads((out/'frozen_manifest.json').read_text())
        if previous!=manifest:
            raise RuntimeError('A different recipe or source version is already frozen in this output directory.')
    (out/'frozen_manifest.json').write_text(json.dumps(manifest,indent=2))
    cache=work/'all_features'
    z=load(cache) if (cache/'features.npz').exists() else build(data,cache,include_heldout=True)
    lab,lock_ids=read_labels(data);labels=lab.set_index('client_id')
    ids=z['ids'];train=np.flatnonzero(np.isin(ids,labels.index));heldout=np.flatnonzero(~np.isin(ids,labels.index))
    if test_only:
        heldout=heldout[~np.isin(ids[heldout],lock_ids)]
    y=np.full(len(ids),-1,dtype=int);y[train]=labels.loc[ids[train],'target_next_recurring_merchant'].map(LABELS.index).values
    p=np.zeros((len(heldout),8))
    for ci,c in enumerate(config['components']):
        wide,candidates=matrices(z,c.get('compact',False));x=wide if c['mode']=='wide' else candidates
        probability,model=fit_predict(x,y,train,heldout,c['mode'],c['iterations'],c['depth'],c['lr'],0)
        p+=c['weight']*probability
        if c['mode'].startswith('lgbm'):
            model.booster_.save_model(str(out/f'component{ci}.txt'))
            importance=model.booster_.feature_importance(importance_type='gain')
        else:
            model.save_model(str(out/f'component{ci}.cbm'))
            importance=model.get_feature_importance(type='PredictionValuesChange')
        if c['mode']!='wide':
            names=matrix_feature_names(z,c.get('compact',False))
            assert len(names)==len(importance)
            pd.DataFrame({'feature':names,'importance':importance}).sort_values('importance',ascending=False).to_csv(out/f'component{ci}_importance.csv',index=False)
    p=adjust(p,config)
    frame=pd.DataFrame(p,index=ids[heldout],columns=LABELS)
    lock_ids=sorted(lock_ids)
    sample=pd.read_csv(Path(data)/'sample_submission.csv')
    if not test_only:
        write_prob(out/'lockbox.csv',lock_ids,frame.loc[lock_ids].to_numpy())
    write_prob(out/'test_proba.csv',sample.client_id,frame.loc[sample.client_id].to_numpy())
    submission=pd.DataFrame({'client_id':sample.client_id,'predicted_next_recurring_merchant':np.array(LABELS)[frame.loc[sample.client_id].to_numpy().argmax(1)]})
    assert submission.client_id.is_unique and submission.client_id.tolist()==sample.client_id.tolist()
    assert set(submission.predicted_next_recurring_merchant)<=set(LABELS)
    submission.to_csv(out/'submission.csv',index=False)
    digest=hashlib.sha256((out/'submission.csv').read_bytes()).hexdigest()
    (out/'submission.sha256').write_text(digest+'  submission.csv\n')
    print('submission',len(submission),'sha256',digest,flush=True)
    if evaluate_lockbox:
        # The only diagnostic read of locked targets in the pipeline.
        locked=pd.read_csv(Path(data)/'valid_labels.csv').set_index('client_id').loc[lock_ids]
        truth=locked.target_next_recurring_merchant.map(LABELS.index).to_numpy()
        prediction=frame.loc[lock_ids].to_numpy().argmax(1)
        f1=confusion_f1(truth,prediction)
        result={'n_clients':len(locked),'macro_f1':float(f1.mean()),'per_class_f1':dict(zip(LABELS,f1.tolist())),'recipe_sha256':manifest['recipe_sha256']}
        (out/'lockbox_metrics.json').write_text(json.dumps(result,indent=2))
        print('one-shot lockbox',json.dumps(result),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['cv','final']);p.add_argument('--data',required=True);p.add_argument('--work',required=True);p.add_argument('--output',required=True);p.add_argument('--recipe',default=str(Path(__file__).with_name('recipe.json')));p.add_argument('--seeds',default='0,1,2');p.add_argument('--evaluate-lockbox',action='store_true');p.add_argument('--test-only',action='store_true');a=p.parse_args()
    config=recipe(a.recipe);work=Path(a.work);out=Path(a.output);work.mkdir(parents=True,exist_ok=True);out.mkdir(parents=True,exist_ok=True)
    if a.stage=='cv':cv(a.data,work,out,config,list(map(int,a.seeds.split(','))))
    else:final(a.data,work,out,config,a.recipe,a.evaluate_lockbox,a.test_only)

if __name__=='__main__':main()
