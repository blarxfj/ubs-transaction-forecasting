"""Assemble repeated-CV component files for the frozen recipe without refitting."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from features import LABELS
from run import recipe,adjust,write_prob
from evaluate import summarize


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--recipe',required=True);p.add_argument('--component',action='append',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    config=recipe(a.recipe);assert len(config['components'])==len(a.component)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    for seed in range(3):
        probabilities=[];reference=None
        for ci,(component,c) in enumerate(zip(a.component,config['components'])):
            frame=pd.read_csv(Path(component)/f'oof_seed{seed}.csv')
            if reference is None:reference=frame[['client_id','fold']]
            else:assert reference.equals(frame[['client_id','fold']])
            probabilities.append(c['weight']*frame[LABELS].values)
        prob=adjust(sum(probabilities),config)
        write_prob(out/f'oof_seed{seed}.csv',reference.client_id,prob,reference.fold)
        if seed==0:write_prob(out/'oof.csv',reference.client_id,prob,reference.fold)
    result=summarize(a.data,[out/f'oof_seed{s}.csv' for s in range(3)])
    (out/'metrics.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':main()
