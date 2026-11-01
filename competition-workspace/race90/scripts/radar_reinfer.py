"""Re-evaluate saved fine-tunes with larger inference ensembles and full test context."""
from pathlib import Path
import argparse,gc,json,time
import numpy as np,pandas as pd,torch
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score,log_loss
from tabicl import TabICLClassifier

p=argparse.ArgumentParser();p.add_argument('--device',default='cuda:0');p.add_argument('--model',required=True);p.add_argument('--estimators',type=int,default=128);args=p.parse_args();torch.set_num_threads(4)
root=Path('/workspace/competition-radar');prior=json.loads((root/(args.model+'.json')).read_text());config=prior['args']
X=pd.read_csv(root/'features_physics_train.csv',index_col=0);Xt=pd.read_csv(root/'features_physics_test.csv',index_col=0);y=pd.read_csv(root/'labels.csv').label.to_numpy();n={'original':79,'all':165,'robust':215,'physics':337}[config['features']];X=X.iloc[:,:n];Xt=Xt.iloc[:,:n]
out=np.zeros((len(y),5));test=np.zeros((len(Xt),5));test_full=np.zeros_like(test);start=time.time();replay=None
old=np.load(root/(args.model+'.npz'),allow_pickle=True)
for fold,(a,b) in enumerate(StratifiedKFold(5,shuffle=True,random_state=config['cv_seed']).split(X,y)):
    path=root/'finetuned_models'/args.model/str(fold)/'best.ckpt'
    if fold==0:
        check=TabICLClassifier(model_path=path,n_estimators=32,device=args.device,random_state=config['seed'],n_jobs=4)
        check.fit(X.iloc[a],y[a]);pr=check.predict_proba(X.iloc[b]);replay={'max_abs_difference':float(np.max(np.abs(pr-old['oof'][b]))),'label_agreement':float(np.mean(pr.argmax(1)==old['oof'][b].argmax(1)))}
        print('CHECKPOINT_REPLAY',replay,flush=True);del check
    m=TabICLClassifier(model_path=path,n_estimators=args.estimators,device=args.device,random_state=config['seed'],n_jobs=4)
    m.fit(X.iloc[a],y[a]);out[b]=m.predict_proba(X.iloc[b]);test+=m.predict_proba(Xt)/5
    # For unlabeled test inference only, use every provided training label as context.
    m.fit(X,y);test_full+=m.predict_proba(Xt)/5
    print('FOLD',fold,accuracy_score(y[b],m.classes_[out[b].argmax(1)]),'seconds',round(time.time()-start,1),flush=True)
    classes=m.classes_;del m;gc.collect();torch.cuda.empty_cache()
name=args.model+f'_infer{args.estimators}'
np.savez_compressed(root/(name+'.npz'),oof=out,test=test,test_full=test_full,classes=classes)
result={'accuracy':accuracy_score(y,classes[out.argmax(1)]),'log_loss':log_loss(y,out),'seconds':time.time()-start,'args':vars(args),'checkpoint_replay':replay,'test_full_note':'Same trained fold weights, all 1000 training samples as test-time labeled context. Outer validation never uses held-out labels.'}
(root/(name+'.json')).write_text(json.dumps(result,indent=2));print('TOTAL',result,flush=True)
