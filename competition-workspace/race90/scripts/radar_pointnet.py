from pathlib import Path
import os,argparse,json,time,copy,random
os.environ['OMP_NUM_THREADS']='4'
import numpy as np,pandas as pd,torch
from torch import nn
from sklearn.preprocessing import QuantileTransformer,LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score,log_loss

parser=argparse.ArgumentParser();parser.add_argument('--device',default='cuda:0');parser.add_argument('--seeds',default='42,123,2026');parser.add_argument('--epochs',type=int,default=160);parser.add_argument('--attention',action='store_true');args=parser.parse_args()
torch.set_num_threads(4)
root=Path('/workspace/competition-radar');device=torch.device(args.device)
le=LabelEncoder();y=le.fit_transform(pd.read_csv(root/'labels.csv').label)
X=pd.read_csv(root/'features_physics_train.csv',index_col=0).to_numpy();Xt=pd.read_csv(root/'features_physics_test.csv',index_col=0).to_numpy()
tr=np.load(root/'pointclouds_train.npz');te=np.load(root/'pointclouds_test.npz');cap=256
def pad_to_cap(data):
    out=np.zeros((len(data['points']),cap,15),np.float32);mask=np.zeros((len(out),cap),bool)
    for i,(p,m) in enumerate(zip(data['points'],data['mask'])):
        p=p[m]
        if len(p)>cap:p=p[np.linspace(0,len(p)-1,cap).astype(int)]
        out[i,:len(p)]=p;mask[i,:len(p)]=True
    return torch.tensor(out,device=device),torch.tensor(mask,device=device)
points,mask=pad_to_cap(tr);tpoints,tmask=pad_to_cap(te)
class CloudClassifier(nn.Module):
    def __init__(self,nfeat):
        super().__init__()
        self.point=nn.Sequential(nn.Linear(15,64),nn.LayerNorm(64),nn.GELU(),nn.Linear(64,64),nn.LayerNorm(64),nn.GELU())
        self.query=nn.Linear(64,1)
        self.attention=nn.TransformerEncoder(nn.TransformerEncoderLayer(64,4,128,dropout=.1,batch_first=True,norm_first=True),1,enable_nested_tensor=False) if args.attention else None
        self.meta=nn.Sequential(nn.Linear(nfeat,128),nn.LayerNorm(128),nn.GELU(),nn.Dropout(.15),nn.Linear(128,64),nn.GELU())
        self.head=nn.Sequential(nn.Linear(64*5,128),nn.LayerNorm(128),nn.GELU(),nn.Dropout(.15),nn.Linear(128,64),nn.GELU(),nn.Dropout(.1),nn.Linear(64,5))
    def forward(self,p,m,x):
        if self.training:
            p=p+torch.randn_like(p)*.005
        z=self.point(p)
        if self.attention is not None:z=self.attention(z,src_key_padding_mask=~m)
        w=m[...,None].float();n=w.sum(1).clamp_min(1)
        mean=(z*w).sum(1)/n;std=(((z-mean[:,None])**2*w).sum(1)/n+1e-5).sqrt()
        maximum=z.masked_fill(~m[...,None],-1e4).max(1).values
        attn=self.query(z).squeeze(-1).masked_fill(~m,-1e4).softmax(1)
        weighted=(z*attn[...,None]).sum(1)
        return self.head(torch.cat([mean,std,maximum,weighted,self.meta(x)],1))
def predict(model,p,m,x):
    model.eval();outputs=[]
    with torch.inference_mode():
        for j in range(0,len(x),128):outputs.append(model(p[j:j+128],m[j:j+128],x[j:j+128]).softmax(-1).cpu().numpy())
    return np.concatenate(outputs)
for seed in map(int,args.seeds.split(',')):
    name=('attention' if args.attention else 'pointnet')+'_s'+str(seed)
    oof=np.zeros((len(y),5));test=np.zeros((len(Xt),5));fold_results=[]
    for fold,(a,b) in enumerate(StratifiedKFold(5,shuffle=True,random_state=42).split(X,y)):
        torch.manual_seed(seed+fold);np.random.seed(seed+fold);random.seed(seed+fold)
        qt=QuantileTransformer(n_quantiles=200,output_distribution='normal',random_state=42)
        xx=torch.tensor(qt.fit_transform(X[a]),dtype=torch.float32,device=device);xv=torch.tensor(qt.transform(X[b]),dtype=torch.float32,device=device);xt=torch.tensor(qt.transform(Xt),dtype=torch.float32,device=device)
        pa,ma=points[a],mask[a];pv,mv=points[b],mask[b];yy=torch.tensor(y[a],device=device)
        model=CloudClassifier(X.shape[1]).to(device);opt=torch.optim.AdamW(model.parameters(),lr=.0015,weight_decay=.02)
        schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,args.epochs,eta_min=.000015)
        best_loss=1e9;best_state=None;best_epoch=0;start=time.time()
        for epoch in range(args.epochs):
            model.train();perm=torch.randperm(len(a),device=device)
            for j in range(0,len(a),64):
                ix=perm[j:j+64];opt.zero_grad(set_to_none=True)
                loss=nn.functional.cross_entropy(model(pa[ix],ma[ix],xx[ix]),yy[ix],label_smoothing=.03)
                loss.backward();nn.utils.clip_grad_norm_(model.parameters(),2);opt.step()
            schedule.step()
            if epoch%5==4 or epoch==args.epochs-1:
                pr=predict(model,pv,mv,xv);vl=log_loss(y[b],pr)
                if vl<best_loss:best_loss=vl;best_epoch=epoch;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                if epoch%40==39:print(name,'fold',fold,'epoch',epoch+1,'acc',accuracy_score(y[b],pr.argmax(1)),'loss',round(vl,4),'time',round(time.time()-start),flush=True)
        model.load_state_dict(best_state);oof[b]=predict(model,pv,mv,xv);test+=predict(model,tpoints,tmask,xt)/5
        dest=root/'pointnet_models';dest.mkdir(exist_ok=True)
        torch.save({'state':best_state,'nfeat':X.shape[1],'classes':le.classes_.tolist(),'best_epoch':best_epoch},dest/f'{name}_fold{fold}.pt')
        fold_results.append({'fold':fold,'accuracy':accuracy_score(y[b],oof[b].argmax(1)),'best_epoch':best_epoch+1,'log_loss':best_loss})
        print(name,fold,fold_results[-1],flush=True)
    np.savez_compressed(root/(name+'.npz'),oof=oof,test=test,classes=le.classes_)
    (root/(name+'.json')).write_text(json.dumps(fold_results,indent=2))
    print(name,'TOTAL',accuracy_score(y,oof.argmax(1)),log_loss(y,oof),flush=True)
