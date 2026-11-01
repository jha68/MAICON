"""Evaluate a distinct tabular foundation model with identical outer folds."""
from pathlib import Path
import argparse, gc, json, time
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import LabelEncoder
from tabdpt import TabDPTClassifier

p = argparse.ArgumentParser()
p.add_argument('--device', default='cuda:0')
p.add_argument('--cv-seed', type=int, default=42)
p.add_argument('--estimators', type=int, default=16)
p.add_argument('--normalizer', default='standard')
p.add_argument('--features', default='original,all,physics')
args = p.parse_args()
torch.set_num_threads(4)
root = Path('/workspace/competition-radar')
X = pd.read_csv(root/'features_physics_train.csv', index_col=0)
Xt = pd.read_csv(root/'features_physics_test.csv', index_col=0)
encoder = LabelEncoder()
y = encoder.fit_transform(pd.read_csv(root/'labels.csv').label)
counts = {'original':79, 'all':165, 'robust':215, 'physics':337}
for feature in args.features.split(','):
    x, xt = X.iloc[:,:counts[feature]].to_numpy(), Xt.iloc[:,:counts[feature]].to_numpy()
    x = np.nan_to_num(x, nan=0., posinf=0., neginf=0.)
    xt = np.nan_to_num(xt, nan=0., posinf=0., neginf=0.)
    out = np.zeros((len(y),5)); test = np.zeros((len(xt),5)); start=time.time(); scores=[]
    name=f'tabdpt_{feature}_{args.normalizer}_cv{args.cv_seed}_n{args.estimators}'
    for fold,(a,b) in enumerate(StratifiedKFold(5,shuffle=True,random_state=args.cv_seed).split(x,y)):
        m = TabDPTClassifier(device=args.device, compile=False, normalizer=args.normalizer, verbose=False)
        m.fit(x[a], y[a])
        query = np.concatenate([x[b], xt])
        probs=m.ensemble_predict_proba(query,n_ensembles=args.estimators,
                                      context_size=None,batch_size=1024,seed=42)
        out[b]=probs[:len(b)];test+=probs[len(b):]/5
        scores.append(accuracy_score(y[b],out[b].argmax(1)))
        print(name,'FOLD',fold,scores[-1],'seconds',round(time.time()-start,1),flush=True)
        del m;gc.collect();torch.cuda.empty_cache()
    result={'accuracy':accuracy_score(y,out.argmax(1)), 'log_loss':log_loss(y,out),
            'fold_accuracies':scores,'seconds':time.time()-start,'args':vars(args)}
    np.savez_compressed(root/(name+'.npz'),oof=out,test=test,classes=encoder.classes_)
    (root/(name+'.json')).write_text(json.dumps(result,indent=2))
    print(name,'TOTAL',result,flush=True)
