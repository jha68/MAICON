from pathlib import Path
import argparse,json,time
import numpy as np,pandas as pd,torch
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score,log_loss
from tabicl import TabICLClassifier

p=argparse.ArgumentParser();p.add_argument('--device',default='cuda:2');p.add_argument('--estimators',type=int,default=64);p.add_argument('--seed',type=int,default=57);p.add_argument('--cv-seed',type=int,default=42);args=p.parse_args()
torch.set_num_threads(4)
root=Path('/workspace/competition-radar');y=pd.read_csv(root/'labels.csv').label.to_numpy()
X=pd.read_csv(root/'features_physics_train.csv',index_col=0);Xt=pd.read_csv(root/'features_physics_test.csv',index_col=0)
base=list(X.columns[:79]);all_old=list(X.columns[:165])
ego=['ego_speed','ego_full_speed','ego_yaw','ego_residual','ego_full_residual','ols_trans_speed','robust_trans_speed','frame_trans_speed']
motion=[c for c in X if c.startswith(('absolute_joint','ols_movement','robust_movement'))]
robust=[c for c in X.columns[165:215] if not c.startswith(('compvel','rawabsvel'))]
less=[c for c in base if not c.startswith(('range_m','azimuth_rad','sensor_')) and not (c.startswith(('x_m','y_m')) and c.endswith(('mean','median','min','max','q10','q90')))]
options=[('base_ego',base+ego),('base_ego_motion',base+ego+motion),('all_ego',all_old+ego),('less_ego',less+ego),('base_robust_ego',base+robust+ego),('base_ego_speed',base+['ego_speed','robust_trans_speed'])]
options += [('select_'+str(k),k) for k in [12,20,32,48,64,96,160]]
options += [('base_outlier2',base),('base_outlier8',base),('base_seed',base),('all_seed',all_old)]
for feature,spec in options:
    name=feature+'_s'+str(args.seed)+'_cv'+str(args.cv_seed)
    out=np.zeros((len(y),5));test=np.zeros((len(Xt),5));start=time.time();selected=[]
    for fold,(a,b) in enumerate(StratifiedKFold(5,shuffle=True,random_state=args.cv_seed).split(X,y)):
        if isinstance(spec,int):
            select=ExtraTreesClassifier(n_estimators=350,max_features=.8,min_samples_leaf=2,n_jobs=4,random_state=args.seed)
            select.fit(X.iloc[a],y[a]);columns=list(X.columns[np.argsort(select.feature_importances_)[-spec:]])
        else:columns=spec
        selected.append(columns)
        threshold=2 if feature.endswith('outlier2') else 8 if feature.endswith('outlier8') else 4
        m=TabICLClassifier(n_estimators=args.estimators,device=args.device,random_state=args.seed,n_jobs=4,outlier_threshold=threshold)
        m.fit(X.iloc[a][columns],y[a]);out[b]=m.predict_proba(X.iloc[b][columns]);test+=m.predict_proba(Xt[columns])/5
        print(name,fold,accuracy_score(y[b],m.classes_[out[b].argmax(1)]),'seconds',round(time.time()-start,1),flush=True)
    np.savez_compressed(root/(name+'.npz'),oof=out,test=test,classes=m.classes_)
    (root/(name+'_features.json')).write_text(json.dumps(selected,indent=2))
    print(name,'TOTAL',accuracy_score(y,m.classes_[out.argmax(1)]),log_loss(y,out),flush=True)
