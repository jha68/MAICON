"""Mitra comparison in an isolated dependency environment."""
from pathlib import Path
import argparse,gc,json,time
import numpy as np,pandas as pd,torch
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score,log_loss
from autogluon.tabular.models.mitra.sklearn_interface import MitraClassifier

p=argparse.ArgumentParser();p.add_argument('--device',default='cuda:1');p.add_argument('--features',default='original,all');p.add_argument('--finetune',action='store_true');p.add_argument('--steps',type=int,default=50);p.add_argument('--estimators',type=int,default=4);p.add_argument('--lr',type=float,default=3e-5);args=p.parse_args();torch.set_num_threads(4)
root=Path('/workspace/competition-radar');X=pd.read_csv(root/'features_physics_train.csv',index_col=0);Xt=pd.read_csv(root/'features_physics_test.csv',index_col=0);enc=LabelEncoder();y=enc.fit_transform(pd.read_csv(root/'labels.csv').label)
for feature in args.features.split(','):
    n={'original':79,'all':165,'physics':337}[feature];x=X.iloc[:,:n].to_numpy(np.float32);xt=Xt.iloc[:,:n].to_numpy(np.float32)
    out=np.zeros((len(y),5));test=np.zeros((len(xt),5));start=time.time();stats=[];name=f'mitra_{feature}_ft{int(args.finetune)}_n{args.estimators}_lr{args.lr:g}_cv42'
    for fold,(a,b) in enumerate(StratifiedKFold(5,shuffle=True,random_state=42).split(x,y)):
        m=MitraClassifier(device=args.device,fine_tune=args.finetune,fine_tune_steps=args.steps,n_estimators=args.estimators,
             lr=args.lr,warmup_steps=5,patience=15,seed=42,shuffle_classes=True,shuffle_features=True,verbose=False)
        # Internal train/validation splitting sees only the outer training subset.
        m.fit(x[a],y[a],time_limit=180 if args.finetune else None)
        pr=m.predict_proba(np.concatenate([x[b],xt]));assert pr.shape==(len(b)+len(xt),5),pr.shape
        out[b]=pr[:len(b)];test+=pr[len(b):]/5
        if args.finetune:
            dest=root/'mitra_finetuned_models'/name;dest.mkdir(parents=True,exist_ok=True)
            for j,trainer in enumerate(m.trainers):
                torch.save({'state_dict':trainer.model.state_dict(),'outer_train_indices':a.tolist(),
                            'classes':enc.classes_.tolist(),'args':vars(args)},dest/f'fold{fold}_model{j}.pt')
        stats.append({'fold':fold,'accuracy':accuracy_score(y[b],out[b].argmax(1))});print(name,stats[-1],'seconds',round(time.time()-start,1),flush=True)
        del m;gc.collect();torch.cuda.empty_cache()
    np.savez_compressed(root/(name+'.npz'),oof=out,test=test,classes=enc.classes_)
    result={'accuracy':accuracy_score(y,out.argmax(1)),'log_loss':log_loss(y,out),'folds':stats,'seconds':time.time()-start,'args':vars(args),'validation':'Outer-fold labels excluded from model fitting and internal validation.'}
    (root/(name+'.json')).write_text(json.dumps(result,indent=2));print('TOTAL',result,flush=True)
