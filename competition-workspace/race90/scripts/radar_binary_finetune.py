"""Fine-tuned pedestrian/group specialist with outer-fold isolation."""
from pathlib import Path
import argparse,gc,json,time
import numpy as np,pandas as pd,torch
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score,log_loss
from tabicl import FinetunedTabICLClassifier,TabICLClassifier
p=argparse.ArgumentParser();p.add_argument('--device',default='cuda:0');p.add_argument('--features',default='original');p.add_argument('--cv-seed',type=int,default=42);args=p.parse_args()
torch.set_num_threads(4);root=Path('/workspace/competition-radar')
n={'original':79,'all':165,'physics':337}[args.features]
X=pd.read_csv(root/'features_physics_train.csv',index_col=0).iloc[:,:n];Xt=pd.read_csv(root/'features_physics_test.csv',index_col=0).iloc[:,:n];y=pd.read_csv(root/'labels.csv').label.to_numpy()
pair=np.array(['PERSONNEL','PERSONNEL_GROUP']);mask=np.isin(y,pair)
oof=np.zeros((len(y),2));test=np.zeros((len(Xt),2));test_full=np.zeros_like(test)
name=f'binary_ft_{args.features}_cv{args.cv_seed}';start=time.time()
for fold,(a,b) in enumerate(StratifiedKFold(5,shuffle=True,random_state=args.cv_seed).split(X,y)):
 aa=a[mask[a]];folder=root/'binary_finetuned_models'/name/str(fold);folder.mkdir(parents=True,exist_ok=True)
 m=FinetunedTabICLClassifier(epochs=100,learning_rate=3e-5,n_estimators_finetune=4,n_estimators_validation=8,n_estimators_inference=128,validation_split_ratio=.15,early_stopping=True,patience=15,eval_metric='log_loss',device=args.device,random_state=42,save_interval=0,verbose=True,extra_classifier_kwargs={'n_jobs':4})
 m.fit(X.iloc[aa],y[aa],output_dir=folder);assert np.array_equal(m.classes_,pair)
 oof[b]=m.predict_proba(X.iloc[b]);test+=m.predict_proba(Xt)/5;del m;gc.collect();torch.cuda.empty_cache()
 m=TabICLClassifier(model_path=folder/'best.ckpt',n_estimators=128,device=args.device,random_state=42,n_jobs=4)
 m.fit(X[mask],y[mask]);test_full+=m.predict_proba(Xt)/5;del m;gc.collect();torch.cuda.empty_cache()
 bb=b[mask[b]];print('FOLD',fold,'binary accuracy',accuracy_score(y[bb],pair[oof[bb].argmax(1)]),'seconds',round(time.time()-start,1),flush=True)
np.savez_compressed(root/(name+'.npz'),oof=oof,test=test,test_full=test_full,classes=pair)
result={'binary_accuracy':accuracy_score(y[mask],pair[oof[mask].argmax(1)]),'log_loss':log_loss(y[mask],oof[mask]),'seconds':time.time()-start,'args':vars(args),'validation':'Outer-held-out labels excluded from training and inner early stopping; score limited to PERSONNEL/PERSONNEL_GROUP.'}
(root/(name+'.json')).write_text(json.dumps(result,indent=2));print('TOTAL',result,flush=True)
