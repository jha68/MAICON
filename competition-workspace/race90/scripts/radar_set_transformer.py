"""Raw radar-set attention + metadata, with inner stopping and outer refit."""
from pathlib import Path
import argparse,gc,json,time
import numpy as np,pandas as pd,torch
from torch import nn
from sklearn.preprocessing import QuantileTransformer,LabelEncoder
from sklearn.model_selection import StratifiedKFold,train_test_split
from sklearn.metrics import accuracy_score,log_loss

p=argparse.ArgumentParser();p.add_argument('--device',default='cuda:2');p.add_argument('--epochs',type=int,default=160);p.add_argument('--seed',type=int,default=42);args=p.parse_args()
torch.set_num_threads(4);device=torch.device(args.device);root=Path('/workspace/competition-radar')
enc=LabelEncoder();y=enc.fit_transform(pd.read_csv(root/'labels.csv').label)
X=pd.read_csv(root/'features_physics_train.csv',index_col=0).iloc[:,:79].to_numpy();Xt=pd.read_csv(root/'features_physics_test.csv',index_col=0).iloc[:,:79].to_numpy()
def load_cloud(path):
    d=np.load(path);out=np.zeros((len(d['points']),128,15),np.float32);mask=np.zeros(out.shape[:2],bool)
    for i,(pts,valid) in enumerate(zip(d['points'],d['mask'])):
        pts=pts[valid]
        if len(pts)>128:pts=pts[np.linspace(0,len(pts)-1,128,dtype=int)]
        out[i,:len(pts)]=pts;mask[i,:len(pts)]=True
    return torch.tensor(out,device=device),torch.tensor(mask,device=device)
cloud,mask=load_cloud(root/'pointclouds_train.npz');tcloud,tmask=load_cloud(root/'pointclouds_test.npz')
class Model(nn.Module):
    def __init__(self):
        super().__init__();self.input=nn.Sequential(nn.Linear(15,64),nn.LayerNorm(64),nn.GELU())
        self.attn=nn.TransformerEncoder(nn.TransformerEncoderLayer(64,4,128,.15,batch_first=True,norm_first=True),2,enable_nested_tensor=False)
        self.meta=nn.Sequential(nn.Linear(79,128),nn.LayerNorm(128),nn.GELU(),nn.Dropout(.15),nn.Linear(128,64))
        self.head=nn.Sequential(nn.Linear(64*4,128),nn.LayerNorm(128),nn.GELU(),nn.Dropout(.2),nn.Linear(128,5))
    def forward(self,p,m,x):
        if self.training:p=p+torch.randn_like(p)*.015
        z=self.attn(self.input(p),src_key_padding_mask=~m);w=m[...,None].float();n=w.sum(1).clamp_min(1)
        mean=(z*w).sum(1)/n;std=(((z-mean[:,None])**2*w).sum(1)/n+1e-5).sqrt();mx=z.masked_fill(~m[...,None],-1e4).max(1).values
        return self.head(torch.cat([mean,std,mx,self.meta(x)],-1))
def predict(m,p,valid,x):
    m.eval()
    with torch.inference_mode():return torch.cat([m(p[i:i+128],valid[i:i+128],x[i:i+128]).softmax(-1) for i in range(0,len(x),128)]).cpu().numpy()
def train(m,indices,x,epochs,validation=None):
    opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.02);yy=torch.tensor(y[indices],device=device);best=1e9;best_epoch=20;stale=0
    for epoch in range(epochs):
        m.train();order=torch.randperm(len(indices),device=device)
        for ix in order.split(64):
            opt.zero_grad(set_to_none=True);loss=nn.functional.cross_entropy(m(cloud[indices][ix],mask[indices][ix],x[ix]),yy[ix],label_smoothing=.02)
            loss.backward();nn.utils.clip_grad_norm_(m.parameters(),2);opt.step()
        if validation is not None and epoch%5==4:
            vi,vx=validation;score=log_loss(y[vi],predict(m,cloud[vi],mask[vi],vx))
            if score<best:best=score;best_epoch=epoch+1;stale=0
            else:stale+=5
            if stale>=30:break
    return best_epoch if validation is not None else epochs
oof=np.zeros((len(y),5));test=np.zeros((len(Xt),5));stats=[];start=time.time()
for fold,(a,b) in enumerate(StratifiedKFold(5,shuffle=True,random_state=42).split(X,y)):
    ia,iv=train_test_split(a,test_size=.2,stratify=y[a],random_state=args.seed)
    def transform(fit,other):
        q=QuantileTransformer(n_quantiles=200,output_distribution='normal',random_state=42)
        xx=torch.tensor(q.fit_transform(X[fit]),dtype=torch.float32,device=device)
        return xx,[torch.tensor(q.transform(v),dtype=torch.float32,device=device) for v in other]
    torch.manual_seed(args.seed+fold);xx,[xv]=transform(ia,[X[iv]]);m=Model().to(device);epochs=train(m,ia,xx,args.epochs,(iv,xv));del m
    torch.manual_seed(args.seed+fold);xx,[xv,xt]=transform(a,[X[b],Xt]);m=Model().to(device);train(m,a,xx,epochs)
    oof[b]=predict(m,cloud[b],mask[b],xv);test+=predict(m,tcloud,tmask,xt)/5
    stats.append({'fold':fold,'epochs':epochs,'accuracy':accuracy_score(y[b],oof[b].argmax(1))});print(stats[-1],'seconds',round(time.time()-start,1),flush=True)
    del m;gc.collect();torch.cuda.empty_cache()
name=f'set_transformer_s{args.seed}';np.savez_compressed(root/(name+'.npz'),oof=oof,test=test,classes=enc.classes_)
result={'accuracy':accuracy_score(y,oof.argmax(1)),'log_loss':log_loss(y,oof),'folds':stats,'seconds':time.time()-start,'validation':'Outer labels not used for training or stopping. Inner holdout chooses epochs, then fresh model refits outer training set.'}
(root/(name+'.json')).write_text(json.dumps(result,indent=2));print('TOTAL',result,flush=True)
