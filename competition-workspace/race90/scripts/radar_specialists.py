"""Binary specialists trained only on the training side of each outer fold."""
from pathlib import Path
import argparse,itertools,json,time
import numpy as np,pandas as pd,torch
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score,log_loss
from tabicl import TabICLClassifier
p=argparse.ArgumentParser();p.add_argument('--device',default='cuda:1');p.add_argument('--cv-seed',default=42,type=int);args=p.parse_args()
torch.set_num_threads(4)
root=Path('/workspace/competition-radar')
X=pd.read_csv(root/'features_physics_train.csv',index_col=0);Xt=pd.read_csv(root/'features_physics_test.csv',index_col=0);y=pd.read_csv(root/'labels.csv').label.to_numpy()
classes=np.unique(y)
pairs=[(2,3),(0,1),(1,4),(2,4),(3,4)]
base_oof=(np.load(root/'tabicl_original.npz',allow_pickle=True)['oof']+np.load(root/'tabicl_all.npz',allow_pickle=True)['oof'])/2
base_test=(np.load(root/'tabicl_original.npz',allow_pickle=True)['test']+np.load(root/'tabicl_all.npz',allow_pickle=True)['test'])/2
for feature in ['original','all','physics','select32']:
    oof=np.zeros((len(y),len(pairs),2));test=np.zeros((len(Xt),len(pairs),2));start=time.time()
    for fold,(a,b) in enumerate(StratifiedKFold(5,shuffle=True,random_state=args.cv_seed).split(X,y)):
        for pi,(ci,cj) in enumerate(pairs):
            aa=a[np.isin(y[a],classes[[ci,cj]])]
            columns=list(X.columns[:79 if feature=='original' else 165 if feature=='all' else 337])
            if feature=='select32':
                selector=ExtraTreesClassifier(n_estimators=300,min_samples_leaf=2,n_jobs=4,random_state=42).fit(X.iloc[aa],y[aa])
                columns=list(X.columns[np.argsort(selector.feature_importances_)[-32:]])
            m=TabICLClassifier(n_estimators=32,device=args.device,random_state=42,n_jobs=4)
            m.fit(X.iloc[aa][columns],y[aa]);oof[b,pi]=m.predict_proba(X.iloc[b][columns]);test[:,pi]+=m.predict_proba(Xt[columns])/5
        print(feature,fold,round(time.time()-start,1),flush=True)
    name=f'specialists_{feature}_cv{args.cv_seed}'
    np.savez_compressed(root/(name+'.npz'),oof=oof,test=test,classes=classes,pairs=pairs)
    # Replace only the relative probabilities of the most frequently confused pairs.
    if args.cv_seed==42:
        for weight in [.25,.5,.75,1]:
            pred=base_oof.copy()
            for pi,(ci,cj) in enumerate(pairs[:2]):
                mass=pred[:,[ci,cj]].sum(1,keepdims=True)
                pred[:,[ci,cj]]=(1-weight)*pred[:,[ci,cj]]+weight*mass*oof[:,pi]
            print(name,weight,accuracy_score(y,classes[pred.argmax(1)]),log_loss(y,pred),flush=True)
