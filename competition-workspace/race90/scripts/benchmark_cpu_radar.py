"""Measure real two-core CPU refitting on the original outer folds."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[k]='2'
os.environ['CUDA_VISIBLE_DEVICES']=''
if hasattr(os,'sched_getaffinity'):
    os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:2]))
from pathlib import Path
import time,json,resource
import numpy as np,pandas as pd
from sklearn.ensemble import ExtraTreesClassifier,HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

root=Path('/workspace/competition-radar')
out=root/'cpu_retrain_comparison';out.mkdir(exist_ok=True)
X=pd.read_csv(root/'features_improved.csv',index_col=0)
Xt=pd.read_csv(root/'features_test_improved.csv',index_col=0)
y=pd.read_csv(root/'labels.csv').label.to_numpy()
classes=np.unique(y)
folds=list(StratifiedKFold(5,shuffle=True,random_state=42).split(X,y))
specs=[
 ('extra_trees',lambda:ExtraTreesClassifier(n_estimators=500,max_features=1.0,n_jobs=2,random_state=42)),
 ('hist_gradient',lambda:HistGradientBoostingClassifier(max_iter=300,learning_rate=.06,max_leaf_nodes=15,l2_regularization=3,early_stopping=False,random_state=42)),
]
try:
    from catboost import CatBoostClassifier
    specs.append(('catboost',lambda:CatBoostClassifier(iterations=1000,depth=4,learning_rate=.035,l2_leaf_reg=5,loss_function='MultiClass',thread_count=2,verbose=False,random_seed=42,allow_writing_files=False)))
except ImportError:
    pass
report=[]
for name,make in specs:
    start=time.perf_counter();oof=np.zeros((len(y),len(classes)))
    for fold,(a,b) in enumerate(folds):
        m=make();m.fit(X.iloc[a],y[a]);oof[b]=m.predict_proba(X.iloc[b])
        print(name,'fold',fold,'accuracy',accuracy_score(y[b],classes[oof[b].argmax(1)]),'seconds',round(time.perf_counter()-start,2),flush=True)
    cv_seconds=time.perf_counter()-start
    start=time.perf_counter();m=make();m.fit(X,y);test=m.predict_proba(Xt)
    item={'model':name,'accuracy':accuracy_score(y,classes[oof.argmax(1)]),'delta_percentage_points_vs_893':round((accuracy_score(y,classes[oof.argmax(1)])-.893)*100,2),'cv_seconds':cv_seconds,'full_fit_and_predict_seconds':time.perf_counter()-start,'cpu_affinity':sorted(os.sched_getaffinity(0)),'max_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    np.savez_compressed(out/(name+'.npz'),oof=oof,test=test,classes=classes)
    pd.DataFrame({'sample_id':Xt.index,'label':classes[test.argmax(1)]}).to_csv(out/(name+'.csv'),index=False)
    report.append(item);(out/'report.json').write_text(json.dumps(report,indent=2))
    print('RESULT',json.dumps(item),flush=True)
print('CPU_RADAR_COMPARISON_DONE',flush=True)
