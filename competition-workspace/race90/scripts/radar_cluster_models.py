from pathlib import Path
import argparse,time,json
import numpy as np,pandas as pd,torch
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score,log_loss
from tabicl import TabICLClassifier
from lightgbm import LGBMClassifier

p=argparse.ArgumentParser();p.add_argument('--device',default='cuda:1');args=p.parse_args();torch.set_num_threads(4)
root=Path('/workspace/competition-radar');X=pd.read_csv(root/'features_physics_train.csv',index_col=0);Xt=pd.read_csv(root/'features_physics_test.csv',index_col=0);y=pd.read_csv(root/'labels.csv').label.to_numpy()
extra=pd.read_csv(root/'cluster_features_train.csv',index_col=0);et=pd.read_csv(root/'cluster_features_test.csv',index_col=0);X=pd.concat([X,extra],axis=1);Xt=pd.concat([Xt,et],axis=1)
base=list(X.columns[:79]);old=list(X.columns[:165]);new=list(extra.columns)
sets={'base_cluster':base+new,'base_cluster_sensor':base+[c for c in new if c.startswith('cluster_sensor')],
      'base_cluster_rcsv':base+[c for c in new if c.startswith('cluster_rcsv') or c.startswith('cluster_spacetime')],
      'all_cluster':old+new}
for family in ['tabicl','lightgbm']:
    for feature,columns in sets.items():
        name=f'{family}_{feature}_cv42';oof=np.zeros((len(y),5));test=np.zeros((len(Xt),5));start=time.time()
        for fold,(a,b) in enumerate(StratifiedKFold(5,shuffle=True,random_state=42).split(X,y)):
            m=TabICLClassifier(n_estimators=64,device=args.device,random_state=42,n_jobs=4) if family=='tabicl' else LGBMClassifier(n_estimators=1000,learning_rate=.025,num_leaves=15,min_child_samples=15,colsample_bytree=.8,reg_lambda=2.,verbosity=-1,n_jobs=8,random_state=42)
            m.fit(X.iloc[a][columns],y[a]);oof[b]=m.predict_proba(X.iloc[b][columns]);test+=m.predict_proba(Xt[columns])/5
            print(name,fold,accuracy_score(y[b],m.classes_[oof[b].argmax(1)]),'seconds',round(time.time()-start,1),flush=True)
        np.savez_compressed(root/(name+'.npz'),oof=oof,test=test,classes=m.classes_)
        result={'accuracy':accuracy_score(y,m.classes_[oof.argmax(1)]),'log_loss':log_loss(y,oof),'seconds':time.time()-start,'features':columns,'family':family}
        (root/(name+'.json')).write_text(json.dumps(result,indent=2));print(name,'TOTAL',result['accuracy'],result['log_loss'],flush=True)
