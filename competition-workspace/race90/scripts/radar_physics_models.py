from pathlib import Path
import argparse,json,time
import numpy as np,pandas as pd,torch
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score,log_loss
from tabicl import TabICLClassifier

p=argparse.ArgumentParser();p.add_argument('--device',default='cuda:2');p.add_argument('--estimators',type=int,default=64);p.add_argument('--seed',type=int,default=42);p.add_argument('--cv-seed',type=int,default=42);args=p.parse_args()
torch.set_num_threads(4)
root=Path('/workspace/competition-radar');y=pd.read_csv(root/'labels.csv').label.to_numpy()
X=pd.read_csv(root/'features_physics_train.csv',index_col=0);Xt=pd.read_csv(root/'features_physics_test.csv',index_col=0)
physics=list(X.columns[215:]);base=list(X.columns[:79])
compact=[c for c in base if not c.startswith(('range_m','azimuth_rad','x_m_','y_m_','radial_velocity_mps','sensor_'))]
for feature,columns in [('physics_all',list(X.columns)),('physics_base',base+physics),('physics_compact',compact+physics)]:
    name=feature+'_s'+str(args.seed)+'_cv'+str(args.cv_seed)
    out=np.zeros((len(y),5));test=np.zeros((len(Xt),5));start=time.time()
    for fold,(a,b) in enumerate(StratifiedKFold(5,shuffle=True,random_state=args.cv_seed).split(X,y)):
        m=TabICLClassifier(n_estimators=args.estimators,device=args.device,random_state=args.seed,n_jobs=4)
        m.fit(X.iloc[a][columns],y[a]);out[b]=m.predict_proba(X.iloc[b][columns]);test+=m.predict_proba(Xt[columns])/5
        print(name,fold,accuracy_score(y[b],m.classes_[out[b].argmax(1)]),'seconds',time.time()-start,flush=True)
    np.savez_compressed(root/(name+'.npz'),oof=out,test=test,classes=m.classes_)
    print(name,'TOTAL',accuracy_score(y,m.classes_[out.argmax(1)]),log_loss(y,out),flush=True)
