"""Transductive context experiment: only confident model predictions supply pseudo-labels."""
from pathlib import Path
import argparse,gc,time,json
import numpy as np,pandas as pd,torch
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score,log_loss
from tabicl import TabICLClassifier

p=argparse.ArgumentParser();p.add_argument('--device',default='cuda:1');p.add_argument('--cv-seed',type=int,default=42);p.add_argument('--thresholds',default='.95,.99');args=p.parse_args();torch.set_num_threads(4)
root=Path('/workspace/competition-radar');X=pd.read_csv(root/'features_physics_train.csv',index_col=0);Xt=pd.read_csv(root/'features_physics_test.csv',index_col=0);y=pd.read_csv(root/'labels.csv').label.to_numpy()
models=[(f'finetune_original_lr3e-05_s42_cv{args.cv_seed}_e120',79),(f'finetune_all_lr3e-05_s42_cv{args.cv_seed}_e100',165)]
saved=[np.load(root/(name+'_infer128.npz'),allow_pickle=True) for name,n in models];classes=saved[0]['classes']
teacher=np.mean([r['oof'] for r in saved],axis=0);tt=np.mean([r['test_full'] for r in saved],axis=0)
for threshold in map(float,args.thresholds.split(',')):
    out=teacher.copy();test=tt.copy();start=time.time();tconf=np.where(tt.max(1)>=threshold)[0];tquery=np.where(tt.max(1)<threshold)[0];test[tquery]=0
    for fold,(a,b) in enumerate(StratifiedKFold(5,shuffle=True,random_state=args.cv_seed).split(X,y)):
        confident=b[teacher[b].max(1)>=threshold];query=b[teacher[b].max(1)<threshold];out[query]=0
        for name,n in models:
            m=TabICLClassifier(model_path=root/'finetuned_models'/name/str(fold)/'best.ckpt',n_estimators=128,device=args.device,random_state=42,n_jobs=4)
            # Ground-truth y[b] is used only when computing the final score.
            xx=np.concatenate([X.iloc[a,:n].to_numpy(),X.iloc[confident,:n].to_numpy()]);yy=np.concatenate([y[a],classes[teacher[confident].argmax(1)]])
            m.fit(xx,yy)
            if len(query):out[query]+=m.predict_proba(X.iloc[query,:n].to_numpy())/2
            xx=np.concatenate([X.iloc[:,:n].to_numpy(),Xt.iloc[tconf,:n].to_numpy()]);yy=np.concatenate([y,classes[tt[tconf].argmax(1)]])
            m.fit(xx,yy)
            if len(tquery):test[tquery]+=m.predict_proba(Xt.iloc[tquery,:n].to_numpy())/10
            del m;gc.collect();torch.cuda.empty_cache()
        print(threshold,'fold',fold,'pseudo_context',len(confident),'uncertain',len(query),'accuracy',accuracy_score(y[b],classes[out[b].argmax(1)]),'seconds',round(time.time()-start,1),flush=True)
    name=f'pseudo_context_t{threshold:g}_cv{args.cv_seed}';out/=out.sum(1,keepdims=True);test/=test.sum(1,keepdims=True)
    np.savez_compressed(root/(name+'.npz'),oof=out,test=test,classes=classes)
    result={'accuracy':accuracy_score(y,classes[out.argmax(1)]),'teacher_accuracy':accuracy_score(y,classes[teacher.argmax(1)]),'log_loss':log_loss(y,out),'cv_seed':args.cv_seed,'threshold':threshold,'test_pseudo_labels':len(tconf),'seconds':time.time()-start,'validation':'Unlabeled outer-validation features and model pseudo-labels used as transductive context; no outer-validation ground-truth labels used for inference.'}
    (root/(name+'.json')).write_text(json.dumps(result,indent=2));print('TOTAL',result,flush=True)
