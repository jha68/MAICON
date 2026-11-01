from pathlib import Path
import numpy as np,pandas as pd,json,os
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from tabicl import TabICLClassifier
root=Path('/workspace/competition-radar');X=pd.read_csv(root/'features_improved.csv',index_col=0);Xt=pd.read_csv(root/'features_test_improved.csv',index_col=0);y=pd.read_csv(root/'labels.csv').label.to_numpy()
cols=list(X.columns[:79])
for name,columns in [('original',cols),('all',list(X.columns))]:
 oof=np.zeros((len(y),5));testpred=np.zeros((len(Xt),5))
 for fold,(a,b) in enumerate(StratifiedKFold(5,shuffle=True,random_state=42).split(X,y)):
  m=TabICLClassifier(n_estimators=16,device='cuda:2',random_state=42,n_jobs=4)
  m.fit(X.iloc[a][columns],y[a]);oof[b]=m.predict_proba(X.iloc[b][columns]);testpred+=m.predict_proba(Xt[columns])/5
  print(name,fold,accuracy_score(y[b],m.classes_[oof[b].argmax(1)]),flush=True)
 np.savez_compressed(root/('tabicl_'+name+'.npz'),oof=oof,test=testpred,classes=m.classes_)
 print(name,'TOTAL',accuracy_score(y,m.classes_[oof.argmax(1)]),flush=True)
