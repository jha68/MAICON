from pathlib import Path
import argparse,json,time,os
os.environ['TABPFN_DISABLE_TELEMETRY']='1'
import numpy as np,pandas as pd,torch
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score,log_loss
from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion

p=argparse.ArgumentParser();p.add_argument('--version',default='v2');p.add_argument('--device',default='cuda:0');p.add_argument('--estimators',type=int,default=32);p.add_argument('--seed',type=int,default=42);p.add_argument('--features',default='original,all,robust');args=p.parse_args()
torch.set_num_threads(4)
root=Path('/workspace/competition-radar');y=pd.read_csv(root/'labels.csv').label.to_numpy()
for feature in args.features.split(','):
    path='features_improved.csv' if feature in ['original','all'] else ('features_robust.csv' if feature=='robust' else 'features_physics_train.csv')
    testpath='features_test_improved.csv' if feature in ['original','all'] else ('features_test_robust.csv' if feature=='robust' else 'features_physics_test.csv')
    X=pd.read_csv(root/path,index_col=0);Xt=pd.read_csv(root/testpath,index_col=0)
    if feature=='original':X=X.iloc[:,:79];Xt=Xt.iloc[:,:79]
    name='tabpfn_'+args.version.replace('.','')+'_'+feature+'_s'+str(args.seed)
    out=np.zeros((len(y),5));test=np.zeros((len(Xt),5));start=time.time()
    for fold,(a,b) in enumerate(StratifiedKFold(5,shuffle=True,random_state=42).split(X,y)):
        m=TabPFNClassifier.create_default_for_version(ModelVersion(args.version),n_estimators=args.estimators,device=args.device,random_state=args.seed,n_preprocessing_jobs=4)
        m.fit(X.iloc[a],y[a]);out[b]=m.predict_proba(X.iloc[b]);test+=m.predict_proba(Xt)/5
        print(name,fold,accuracy_score(y[b],m.classes_[out[b].argmax(1)]),'seconds',time.time()-start,flush=True)
    np.savez_compressed(root/(name+'.npz'),oof=out,test=test,classes=m.classes_)
    print(name,'TOTAL',accuracy_score(y,m.classes_[out.argmax(1)]),log_loss(y,out),flush=True)
