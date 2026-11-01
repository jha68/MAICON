"""Repeated outer cross-validation for model/feature stability."""
from pathlib import Path
import argparse,json,time
import numpy as np,pandas as pd,torch
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score,log_loss
from tabicl import TabICLClassifier
p=argparse.ArgumentParser();p.add_argument('--device',default='cuda:1');p.add_argument('--cv-seed',type=int,default=2026);p.add_argument('--seed',type=int,default=42);p.add_argument('--estimators',type=int,default=32);args=p.parse_args()
torch.set_num_threads(4)
root=Path('/workspace/competition-radar')
X=pd.read_csv(root/'features_physics_train.csv',index_col=0);Xt=pd.read_csv(root/'features_physics_test.csv',index_col=0);y=pd.read_csv(root/'labels.csv').label.to_numpy()
probs=[];start=time.time()
for feature,ncol in [('original',79),('all',165),('robust',215),('physics',337)]:
    oof=np.zeros((len(y),5));test=np.zeros((len(Xt),5));columns=list(X.columns[:ncol])
    for fold,(a,b) in enumerate(StratifiedKFold(5,shuffle=True,random_state=args.cv_seed).split(X,y)):
        m=TabICLClassifier(n_estimators=args.estimators,device=args.device,random_state=args.seed,n_jobs=4)
        m.fit(X.iloc[a][columns],y[a]);oof[b]=m.predict_proba(X.iloc[b][columns]);test+=m.predict_proba(Xt[columns])/5
        print(feature,fold,accuracy_score(y[b],m.classes_[oof[b].argmax(1)]),'seconds',round(time.time()-start,1),flush=True)
    name=f'validate_{feature}_s{args.seed}_cv{args.cv_seed}_n{args.estimators}'
    np.savez_compressed(root/(name+'.npz'),oof=oof,test=test,classes=m.classes_)
    print(name,'TOTAL',accuracy_score(y,m.classes_[oof.argmax(1)]),log_loss(y,oof),flush=True);probs.append(oof)
    print('cumulative_mean',accuracy_score(y,m.classes_[np.mean(probs,axis=0).argmax(1)]),flush=True)
