"""TabM numerical embeddings; inner early stopping, fresh outer-training refit."""
from pathlib import Path
import argparse, gc, json, time
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, QuantileTransformer
from sklearn.metrics import accuracy_score, log_loss
from tabm import TabM
from rtdl_num_embeddings import PiecewiseLinearEmbeddings, compute_bins

p=argparse.ArgumentParser();p.add_argument('--device',default='cuda:0');p.add_argument('--features',default='original,all,physics')
p.add_argument('--seed',type=int,default=42);p.add_argument('--cv-seed',type=int,default=42)
p.add_argument('--embedding',default='ple');p.add_argument('--epochs',type=int,default=250);args=p.parse_args()
torch.set_num_threads(4);device=torch.device(args.device)
root=Path('/workspace/competition-radar');X=pd.read_csv(root/'features_physics_train.csv',index_col=0).to_numpy();Xt=pd.read_csv(root/'features_physics_test.csv',index_col=0).to_numpy()
encoder=LabelEncoder();y=encoder.fit_transform(pd.read_csv(root/'labels.csv').label)

def build(x,seed):
    torch.manual_seed(seed)
    embedding=PiecewiseLinearEmbeddings(compute_bins(x.cpu(),n_bins=16),16,activation=True,version='B') if args.embedding=='ple' else None
    return TabM.make(n_num_features=x.shape[1],d_out=5,num_embeddings=embedding,k=16,n_blocks=3,d_block=256,dropout=.1).to(device)

def prediction(model,x):
    model.eval()
    with torch.inference_mode():
        return torch.cat([model(xx).softmax(-1).mean(1) for xx in x.split(256)]).cpu().numpy()

def train(model,x,y,epochs,validation=None):
    opt=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.0003)
    best=float('inf');best_epoch=0;stale=0
    yy=torch.tensor(y,device=device)
    for epoch in range(epochs):
        model.train();perm=torch.randperm(len(x),device=device)
        for ix in perm.split(128):
            opt.zero_grad(set_to_none=True)
            logits=model(x[ix]);targets=yy[ix,None].expand(-1,logits.shape[1]).reshape(-1)
            loss=torch.nn.functional.cross_entropy(logits.flatten(0,1),targets)
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()
        if validation is not None:
            xv,yv=validation;score=log_loss(yv,prediction(model,xv))
            if score<best-1e-5:best=score;best_epoch=epoch+1;stale=0
            else:stale+=1
            if stale>=25:break
    return best_epoch if validation is not None else epochs

for feature in args.features.split(','):
    n={'original':79,'all':165,'physics':337}[feature];x=X[:,:n];xt=Xt[:,:n]
    out=np.zeros((len(y),5));test=np.zeros((len(xt),5));start=time.time();stats=[]
    name=f'tabm_{feature}_{args.embedding}_s{args.seed}_cv{args.cv_seed}'
    for fold,(a,b) in enumerate(StratifiedKFold(5,shuffle=True,random_state=args.cv_seed).split(x,y)):
        ia,iv=train_test_split(a,test_size=.2,stratify=y[a],random_state=args.seed)
        keep=np.nanstd(x[ia],axis=0)>1e-6
        q=QuantileTransformer(n_quantiles=min(200,len(ia)),output_distribution='normal',random_state=args.seed)
        xx=torch.tensor(q.fit_transform(x[ia][:,keep]),dtype=torch.float32,device=device)
        xv=torch.tensor(q.transform(x[iv][:,keep]),dtype=torch.float32,device=device)
        m=build(xx,args.seed+fold);epochs=train(m,xx,y[ia],args.epochs,(xv,y[iv]));del m
        q=QuantileTransformer(n_quantiles=min(200,len(a)),output_distribution='normal',random_state=args.seed)
        xx=torch.tensor(q.fit_transform(x[a][:,keep]),dtype=torch.float32,device=device)
        xv=torch.tensor(q.transform(x[b][:,keep]),dtype=torch.float32,device=device)
        tt=torch.tensor(q.transform(xt[:,keep]),dtype=torch.float32,device=device)
        m=build(xx,args.seed+fold);train(m,xx,y[a],epochs)
        out[b]=prediction(m,xv);test+=prediction(m,tt)/5
        score=accuracy_score(y[b],out[b].argmax(1));stats.append({'fold':fold,'epochs':epochs,'accuracy':score})
        print(name,stats[-1],'seconds',round(time.time()-start,1),flush=True)
        del m;gc.collect();torch.cuda.empty_cache()
    result={'accuracy':accuracy_score(y,out.argmax(1)),'log_loss':log_loss(y,out),'folds':stats,'seconds':time.time()-start,'args':vars(args),'validation':'Inner split selects epoch count; fresh model refits all outer training data.'}
    np.savez_compressed(root/(name+'.npz'),oof=out,test=test,classes=encoder.classes_)
    (root/(name+'.json')).write_text(json.dumps(result,indent=2));print(name,'TOTAL',result,flush=True)
