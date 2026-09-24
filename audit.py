"""Inspect development data only, excluding locked validation clients."""
import argparse,json,hashlib
from pathlib import Path
import pandas as pd
from features import read_labels


def audit(data):
    labels,locked=read_labels(data);out={'development_clients':len(labels),'lockbox_clients':len(locked),'test_clients':len(pd.read_csv(Path(data)/'sample_submission.csv')),'label_counts':pd.crosstab(labels.target_next_recurring_merchant,labels.source).to_dict(),'splits':{}}
    for split in ['train','valid']:
        tx=pd.read_json(Path(data)/f'{split}_transactions.jsonl',lines=True)
        tx=tx[tx.client_id.isin(labels.client_id)]
        vague=tx.description.isin(['card purchase','digital order','merchant charge','service payment'])
        monthly=tx.description.eq('monthly plan')
        sizes=tx.groupby('client_id').size()
        out['splits'][split]={'clients':tx.client_id.nunique(),'transactions':len(tx),'first_timestamp':str(tx.timestamp.min()),'last_timestamp':str(tx.timestamp.max()),'transactions_per_client_mean':float(sizes.mean()),'transactions_per_client_min':int(sizes.min()),'transactions_per_client_max':int(sizes.max()),'vague_description_count':int(vague.sum()),'vague_description_fraction':float(vague.mean()),'monthly_plan_count':int(monthly.sum()),'monthly_plan_mobile_mcc_fraction':float((tx.loc[monthly,'mcc'].astype(int)==4814).mean())}
    return out

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    result=audit(a.data);Path(a.output).write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
