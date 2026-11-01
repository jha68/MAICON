import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
from pathlib import Path
import json,time,joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.svm import SVC
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import ExtraTreesClassifier,RandomForestClassifier,HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score

root=Path(os.environ.get('RADAR_ROOT','competition/q1'))
X=pd.read_csv(root/'features_improved.csv',index_col=0);Xt=pd.read_csv(root/'features_test_improved.csv',index_col=0)
y=pd.read_csv(root/'labels.csv').label.to_numpy();classes=np.unique(y)
folds=list(StratifiedKFold(5,shuffle=True,random_state=42).split(X,y))
out=root/'model_search';out.mkdir(exist_ok=True)
sets={
 'physical':['count','times','rcs_dbsm_mean','rcs_dbsm_std','comp_abs_mean','radial_velocity_comp_mps_std','spatial_eigen_small','spatial_eigen_large','track_speed'],
 'physics2':['count','times','rcs_dbsm_mean','rcs_dbsm_std','comp_abs_mean','radial_velocity_comp_mps_std','time_eig0','time_eig1','time_xvar','time_yvar','doppler_speed','doppler_resid_std','speed_joint_1','within_eig0','within_eig1','track_speed'],
 'minimal':['count','rcs_dbsm_mean','rcs_dbsm_std','comp_abs_mean','time_eig0','time_eig1'],
 'all':list(X.columns)
}
specs=[]
for features in sets:
 for log in [False,True]:
  for c in [.3,1,3,10,30,100]:
   for gamma in [.03,.1,'scale']:
    specs.append(dict(kind='svm',features=features,log=log,C=c,gamma=gamma))
  if features!='all':
   for reg in [0,.01,.05,.1,.3,.7]:specs.append(dict(kind='qda',features=features,log=log,reg=reg))
   specs.append(dict(kind='gnb',features=features,log=log))
for features in sets:
 for mf in [.5,1.0]:
  specs.append(dict(kind='et',features=features,log=False,max_features=mf))
  specs.append(dict(kind='rf',features=features,log=False,max_features=mf))

def getdata(spec):
 a=X[sets[spec['features']]].to_numpy();b=Xt[sets[spec['features']]].to_numpy()
 if spec['log']:a=np.sign(a)*np.log1p(np.abs(a));b=np.sign(b)*np.log1p(np.abs(b))
 return a,b
def model(spec):
 if spec['kind']=='svm':m=SVC(C=spec['C'],gamma=spec['gamma'],probability=False,random_state=42)
 elif spec['kind']=='qda':m=QuadraticDiscriminantAnalysis(reg_param=spec['reg'])
 elif spec['kind']=='gnb':m=GaussianNB()
 elif spec['kind']=='et':m=ExtraTreesClassifier(n_estimators=500,max_features=spec['max_features'],n_jobs=1,random_state=42)
 elif spec['kind']=='rf':m=RandomForestClassifier(n_estimators=400,max_features=spec['max_features'],n_jobs=1,random_state=42)
 return make_pipeline(StandardScaler(),m)
def evaluate(i,spec):
 a,b=getdata(spec);oof=np.empty(len(y),dtype=object)
 try:
  for tr,va in folds:
   m=model(spec);m.fit(a[tr],y[tr]);oof[va]=m.predict(a[va])
 except (ValueError,np.linalg.LinAlgError) as err:
  print('SKIP',i,str(err),flush=True)
  return dict(index=i,score=-1,spec=spec,error=str(err)),oof
 score=accuracy_score(y,oof)
 print(i,round(score,4),json.dumps(spec),flush=True)
 return dict(index=i,score=score,spec=spec),oof

results=joblib.Parallel(n_jobs=8)(joblib.delayed(evaluate)(i,s) for i,s in enumerate(specs))
results.sort(key=lambda r:r[0]['score'],reverse=True)
(out/'scores.json').write_text(json.dumps([r[0] for r in results],indent=2))
for rank,(rec,oof) in enumerate(results[:8]):
 spec=rec['spec'];a,b=getdata(spec);m=model(spec);m.fit(a,y)
 pd.DataFrame({'sample_id':Xt.index,'label':m.predict(b)}).to_csv(out/f'candidate_{rank}.csv',index=False)
 pd.DataFrame({'sample_id':X.index,'truth':y,'pred':oof}).to_csv(out/f'oof_{rank}.csv',index=False)
 joblib.dump({'model':m,'columns':sets[spec['features']],'log':spec['log'],'spec':spec},out/f'model_{rank}.joblib')
print('BEST',results[0][0],flush=True)
