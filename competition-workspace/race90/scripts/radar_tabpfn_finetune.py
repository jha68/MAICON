"""Fine-tune public TabPFN v2 with an inner validation split, then outer evaluation."""
from pathlib import Path
import argparse,gc,json,os,time
os.environ['TABPFN_DISABLE_TELEMETRY']='1'
import numpy as np,pandas as pd,torch
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score,log_loss
from tabpfn.finetuning import FinetunedTabPFNClassifier
from tabpfn.constants import ModelVersion

p=argparse.ArgumentParser();p.add_argument('--device',default='cuda:0');p.add_argument('--features',default='original');p.add_argument('--lr',type=float,default=1e-5);p.add_argument('--epochs',type=int,default=50);p.add_argument('--seconds-per-fold',type=int,default=180);args=p.parse_args();torch.set_num_threads(4)
root=Path('/workspace/competition-radar');X=pd.read_csv(root/'features_physics_train.csv',index_col=0);Xt=pd.read_csv(root/'features_physics_test.csv',index_col=0);y=pd.read_csv(root/'labels.csv').label.to_numpy();n={'original':79,'all':165}[args.features];X=X.iloc[:,:n];Xt=Xt.iloc[:,:n]
out=np.zeros((len(y),5));test=np.zeros((len(Xt),5));stats=[];start=time.time();name=f'tabpfn_finetune_{args.features}_lr{args.lr:g}_cv42'
for fold,(a,b) in enumerate(StratifiedKFold(5,shuffle=True,random_state=42).split(X,y)):
    m=FinetunedTabPFNClassifier(device=args.device,model_version=ModelVersion.V2,epochs=args.epochs,learning_rate=args.lr,
        time_limit=args.seconds_per_fold,validation_split_ratio=.15,eval_metric='log_loss',early_stopping_patience=10,
        n_estimators_finetune=4,n_estimators_validation=8,n_estimators_final_inference=32,
        save_checkpoint_interval=None,random_state=42,extra_classifier_kwargs={'n_preprocessing_jobs':4})
    dest=root/'tabpfn_finetuned_models'/name/str(fold);dest.mkdir(parents=True,exist_ok=True)
    m.fit(X.iloc[a],y[a],output_dir=dest)
    out[b]=m.predict_proba(X.iloc[b]);test+=m.predict_proba(Xt)/5
    # The fine-tuning wrapper does not expose sklearn's classes_ property.
    # Its inner label encoder uses the sorted labels, and every fold contains all five.
    classes=np.unique(y)
    stats.append({'fold':fold,'accuracy':accuracy_score(y[b],classes[out[b].argmax(1)])});print(name,stats[-1],'seconds',round(time.time()-start,1),flush=True)
    del m;gc.collect();torch.cuda.empty_cache()
np.savez_compressed(root/(name+'.npz'),oof=out,test=test,classes=classes)
result={'accuracy':accuracy_score(y,classes[out.argmax(1)]),'log_loss':log_loss(y,out),'folds':stats,'seconds':time.time()-start,'args':vars(args),'validation':'Outer validation labels never used for tuning or early stopping.'}
(root/(name+'.json')).write_text(json.dumps(result,indent=2));print('TOTAL',result,flush=True)
