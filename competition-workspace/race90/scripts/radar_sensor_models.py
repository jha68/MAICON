from pathlib import Path
import argparse,time
import numpy as np,pandas as pd,torch
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score,log_loss
from tabicl import TabICLClassifier
p=argparse.ArgumentParser();p.add_argument('--device',default='cuda:2');p.add_argument('--cv-seed',type=int,default=42);p.add_argument('--seed',type=int,default=42);args=p.parse_args();torch.set_num_threads(4)
root=Path('/workspace/competition-radar');X=pd.read_csv(root/'features_physics_train.csv',index_col=0);Xt=pd.read_csv(root/'features_physics_test.csv',index_col=0);y=pd.read_csv(root/'labels.csv').label.to_numpy()
extra=pd.read_csv(root/'sensor_features_train.csv',index_col=0);et=pd.read_csv(root/'sensor_features_test.csv',index_col=0);X=pd.concat([X,extra],axis=1);Xt=pd.concat([Xt,et],axis=1)
base=list(X.columns[:79]);old=list(X.columns[:165]);new=list(extra.columns)
shape=[c for c in new if c.startswith('sensorfixed_ols')]+[c for c in new if 'rcs_dbsm' in c]
small=['sensorfixed_ols_trim1_eig0','sensorfixed_ols_trim1_eig1','sensorfixed_ols_offset_spread','sensorfixed_ols_speed','sensorfixed_rcs_dbsm_std','sensorfixed_radial_std','sensorfixed_tangential_std','sensorfixed_angular_std']
for feature,columns in [('base_sensor',base+new),('all_sensor',old+new),('base_sensor_shape',base+shape),('base_sensor_small',base+small),('sensor_physics',base+new+list(X.columns[215:337]))]:
    name=f'{feature}_s{args.seed}_cv{args.cv_seed}';oof=np.zeros((len(y),5));test=np.zeros((len(Xt),5));start=time.time()
    for fold,(a,b) in enumerate(StratifiedKFold(5,shuffle=True,random_state=args.cv_seed).split(X,y)):
        m=TabICLClassifier(n_estimators=32,device=args.device,random_state=args.seed,n_jobs=4)
        m.fit(X.iloc[a][columns],y[a]);oof[b]=m.predict_proba(X.iloc[b][columns]);test+=m.predict_proba(Xt[columns])/5
        print(name,fold,accuracy_score(y[b],m.classes_[oof[b].argmax(1)]),'seconds',round(time.time()-start,1),flush=True)
    np.savez_compressed(root/(name+'.npz'),oof=oof,test=test,classes=m.classes_)
    print(name,'TOTAL',accuracy_score(y,m.classes_[oof.argmax(1)]),log_loss(y,oof),flush=True)
