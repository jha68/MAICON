from pathlib import Path
import numpy as np,pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from tabicl import TabICLClassifier
root=Path('/workspace/competition-radar');X=pd.read_csv(root/'features_robust.csv',index_col=0);Xt=pd.read_csv(root/'features_test_robust.csv',index_col=0);y=pd.read_csv(root/'labels.csv').label.to_numpy()
base=list(X.columns[:79])
physical=[c for c in base if not c.startswith(('range_m_','azimuth_rad_','radial_velocity_mps_')) and not (c.startswith(('x_m_','y_m_')) and any(c.endswith('_'+s) for s in ['mean','min','max','median','q10','q90']))]
robust=[c for c in X if c.startswith(('absvel_','robust_','compvel_','rcs_trim'))]
sets={'compact':physical,'compact_robust':physical+robust,'compact_no_direction':[c for c in physical+robust if not (c.startswith('radial_velocity_comp_mps_') and c.endswith(('mean','median','min','max','q10','q90')))]}
for name,cols in sets.items():
 oof=np.zeros((len(y),5));test=np.zeros((len(Xt),5))
 for fold,(a,b) in enumerate(StratifiedKFold(5,shuffle=True,random_state=42).split(X,y)):
  m=TabICLClassifier(n_estimators=32,device='cuda:0',random_state=43,n_jobs=4)
  m.fit(X.iloc[a][cols],y[a]);oof[b]=m.predict_proba(X.iloc[b][cols]);test+=m.predict_proba(Xt[cols])/5
  print(name,fold,accuracy_score(y[b],m.classes_[oof[b].argmax(1)]),flush=True)
 np.savez_compressed(root/(name+'.npz'),oof=oof,test=test,classes=m.classes_.astype(str))
 print(name,'TOTAL',accuracy_score(y,m.classes_[oof.argmax(1)]),flush=True)
