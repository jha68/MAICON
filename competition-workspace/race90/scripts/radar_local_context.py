"""Context specialists for sparse radar returns; all selection is feature-only."""
from pathlib import Path
import argparse,gc,json,time
import numpy as np,pandas as pd,torch
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from tabicl import TabICLClassifier

p=argparse.ArgumentParser();p.add_argument('--device',default='cuda:2');args=p.parse_args();torch.set_num_threads(4)
root=Path('/workspace/competition-radar');X=pd.read_csv(root/'features_physics_train.csv',index_col=0).iloc[:,:79];Xt=pd.read_csv(root/'features_physics_test.csv',index_col=0).iloc[:,:79];y=pd.read_csv(root/'labels.csv').label.to_numpy();counts=X['count'].to_numpy();tcounts=Xt['count'].to_numpy()
model_name='finetune_original_lr3e-05_s42_cv42_e120';start=time.time()
schemes={'count_overlap':[(0,10,0,25),(10,25,3,80),(25,np.inf,10,np.inf)],
         'count_broad':[(0,15,0,50),(15,np.inf,5,np.inf)]}
for name,groups in schemes.items():
    oof=np.zeros((len(y),5));test=np.zeros((len(Xt),5))
    for fold,(a,b) in enumerate(StratifiedKFold(5,shuffle=True,random_state=42).split(X,y)):
        for lo,hi,clo,chi in groups:
            context=a[(counts[a]>clo)&(counts[a]<=chi)];q=b[(counts[b]>lo)&(counts[b]<=hi)];tq=np.where((tcounts>lo)&(tcounts<=hi))[0]
            m=TabICLClassifier(model_path=root/'finetuned_models'/model_name/str(fold)/'best.ckpt',n_estimators=32,device=args.device,random_state=42,n_jobs=4)
            m.fit(X.iloc[context],y[context]);classes=m.classes_
            assert np.array_equal(classes,np.unique(y))
            if len(q):oof[q]=m.predict_proba(X.iloc[q])
            if len(tq):test[tq]+=m.predict_proba(Xt.iloc[tq])/5
            del m;gc.collect();torch.cuda.empty_cache()
        print(name,'fold',fold,accuracy_score(y[b],classes[oof[b].argmax(1)]),'seconds',round(time.time()-start,1),flush=True)
    np.savez_compressed(root/f'local_context_{name}_cv42.npz',oof=oof,test=test,classes=classes)
    print(name,'TOTAL',accuracy_score(y,classes[oof.argmax(1)]),flush=True)
