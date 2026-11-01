from pathlib import Path
import numpy as np,pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.ensemble import ExtraTreesClassifier
from tabicl import TabICLClassifier
root=Path('/workspace/competition-radar');X=pd.read_csv(root/'features_robust.csv',index_col=0);Xt=pd.read_csv(root/'features_test_robust.csv',index_col=0);y=pd.read_csv(root/'labels.csv').label.to_numpy()
folds=list(StratifiedKFold(5,shuffle=True,random_state=42).split(X,y))
columns=list(X.columns);base=columns[:79];robust=columns[165:]
for name,cols in [('robust_all',columns),('robust_base',base+robust),('extra_trees',columns),('extra_base',base)]:
 oof=np.zeros((len(y),5));test=np.zeros((len(Xt),5))
 for fold,(a,b) in enumerate(folds):
  m=ExtraTreesClassifier(n_estimators=700,max_features=.8,n_jobs=6,random_state=42) if name.startswith('extra') else TabICLClassifier(n_estimators=32,device='cuda:2',random_state=42,n_jobs=4)
  m.fit(X.iloc[a][cols],y[a]);oof[b]=m.predict_proba(X.iloc[b][cols]);test+=m.predict_proba(Xt[cols])/5
  print(name,fold,accuracy_score(y[b],m.classes_[oof[b].argmax(1)]),flush=True)
 np.savez_compressed(root/(name+'.npz'),oof=oof,test=test,classes=m.classes_.astype(str))
 print(name,'TOTAL',accuracy_score(y,m.classes_[oof.argmax(1)]),flush=True)
