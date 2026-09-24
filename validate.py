"""Validate published output schemas and split integrity, without rescoring the lockbox."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from features import LABELS,read_labels,hashed


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',required=True);ap.add_argument('--output',required=True);ap.add_argument('--repeat-submission');a=ap.parse_args();out=Path(a.output)
    labels,locked=read_labels(a.data);sample=pd.read_csv(Path(a.data)/'sample_submission.csv')
    sizes={}
    for filename,expected,has_fold in [('oof.csv',labels.client_id,True),('lockbox.csv',locked,False),('test_proba.csv',sample.client_id,False)]:
        frame=pd.read_csv(out/filename)
        assert frame.columns.tolist()==['client_id']+(['fold'] if has_fold else [])+LABELS
        assert frame.client_id.is_unique and set(frame.client_id)==set(expected)
        prob=frame[LABELS].to_numpy()
        assert np.isfinite(prob).all() and (prob>=0).all() and (prob<=1).all()
        assert np.allclose(prob.sum(1),1,atol=1e-9)
        if has_fold:assert np.array_equal(frame.fold,[hashed(f'0:{c}')%5 for c in frame.client_id])
        sizes[filename]=len(frame)
    submission=pd.read_csv(out/'submission.csv');proba=pd.read_csv(out/'test_proba.csv').set_index('client_id')
    assert submission.columns.tolist()==['client_id','predicted_next_recurring_merchant']
    assert submission.client_id.tolist()==sample.client_id.tolist()
    assert submission.client_id.is_unique
    assert set(submission.predicted_next_recurring_merchant)<=set(LABELS)
    assert np.array_equal(submission.predicted_next_recurring_merchant,np.array(LABELS)[proba.loc[submission.client_id,LABELS].values.argmax(1)])
    result={'output_rows':sizes,'submission_rows':len(submission),'schema_valid':True,'fold_assignments_valid':True,'normalized_probabilities':True,'submission_sha256':hashlib.sha256((out/'submission.csv').read_bytes()).hexdigest()}
    if (out/'metrics.json').exists():
        scores=[]
        for seed in range(3):
            frame=pd.read_csv(out/f'oof_seed{seed}.csv')
            assert frame.client_id.is_unique and set(frame.client_id)==set(labels.client_id)
            assert np.array_equal(frame.fold,[hashed(f'{seed}:{c}')%5 for c in frame.client_id])
            truth=labels.set_index('client_id').loc[frame.client_id,'target_next_recurring_merchant'].to_numpy()
            prediction=np.array(LABELS)[frame[LABELS].values.argmax(1)]
            for fold in range(5):
                mask=frame.fold.values==fold
                scores.append(f1_score(truth[mask],prediction[mask],average='macro',labels=LABELS,zero_division=0))
        reported=json.loads((out/'metrics.json').read_text())['primary_mean_macro_f1']
        difference=abs(float(np.mean(scores))-reported)
        assert difference<1e-12
        result['sklearn_primary_score_agrees']=True
        result['scorer_absolute_difference']=difference
    if a.repeat_submission:
        assert (out/'submission.csv').read_bytes()==Path(a.repeat_submission).read_bytes()
        result['independent_refit_identical_submission']=True
    (out/'results').mkdir(exist_ok=True)
    (out/'results'/'validation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':main()
