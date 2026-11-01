"""Supervised Korean Transformer classification with grouped held-out symptoms."""
from pathlib import Path
import argparse,gc,json,time
import numpy as np,pandas as pd,torch
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score,f1_score
from transformers import AutoTokenizer,AutoModelForSequenceClassification,get_linear_schedule_with_warmup

p=argparse.ArgumentParser();p.add_argument('--device',default='cuda:2');p.add_argument('--epochs',type=int,default=8);p.add_argument('--seed',type=int,default=42);p.add_argument('--model',default='klue/roberta-base');p.add_argument('--tag',default='klue');p.add_argument('--save-models',action='store_true');p.add_argument('--augment-json');args=p.parse_args()
torch.set_num_threads(4);root=Path('/workspace/competition-text');device=torch.device(args.device)
tr=pd.read_csv(root/'train_core.csv');te=pd.read_csv(root/'test_core.csv');enc=LabelEncoder();y=enc.fit_transform(tr.fault_type)
model_id=args.model;tok=AutoTokenizer.from_pretrained(model_id)
def tokens(texts):
    batch=tok(list(texts),padding='max_length',truncation=True,max_length=160,return_tensors='pt')
    return {k:v.to(device) for k,v in batch.items() if k!='token_type_ids'}
full=tokens(tr.report);core=tokens(tr.core);tfull=tokens(te.report);tcore=tokens(te.core)
variants=[full,core]
if args.augment_json:
    augment={r['core']:r['paraphrases'] for r in json.loads(Path(args.augment_json).read_text())}
    assert set(tr.core).issubset(augment)
    variants.extend(tokens([augment[s][j] for s in tr.core]) for j in range(2))
def predict(m,batch):
    m.eval();out=[]
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        for i in range(0,len(batch['input_ids']),64):
            out.append(m(**{k:v[i:i+64] for k,v in batch.items()}).logits.float().softmax(-1).cpu().numpy())
    return np.concatenate(out)

oof=np.zeros((len(tr),6));test=np.zeros((len(te),6));stats=[];start=time.time()
for fold,(a,b) in enumerate(StratifiedGroupKFold(5,shuffle=True,random_state=args.seed).split(tr,y,tr.core)):
    torch.manual_seed(args.seed+fold);np.random.seed(args.seed+fold)
    m=AutoModelForSequenceClassification.from_pretrained(model_id,num_labels=6).to(device)
    opt=torch.optim.AdamW(m.parameters(),lr=2e-5,weight_decay=.01)
    steps=args.epochs*((len(a)+31)//32);schedule=get_linear_schedule_with_warmup(opt,int(steps*.1),steps)
    yy=torch.tensor(y,device=device)
    for epoch in range(args.epochs):
        m.train();indices=np.random.permutation(a)
        for j in range(0,len(indices),32):
            ix=indices[j:j+32]
            if args.augment_json:
                choice=torch.randint(len(variants),(len(ix),),device=device)
                batch={k:torch.stack([v[k][ix] for v in variants],dim=1)[torch.arange(len(ix),device=device),choice] for k in full}
            else:
                use_core=torch.rand(len(ix),1,device=device)<.5
                batch={k:torch.where(use_core,core[k][ix],full[k][ix]) for k in full}
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):loss=m(**batch,labels=yy[ix]).loss
            loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step();schedule.step()
        print('fold',fold,'epoch',epoch+1,'train_loss',round(loss.item(),4),'seconds',round(time.time()-start,1),flush=True)
    va_full=predict(m,{k:v[b] for k,v in full.items()});va_core=predict(m,{k:v[b] for k,v in core.items()})
    oof[b]=(va_full+va_core)/2;test+=(predict(m,tfull)+predict(m,tcore))/10
    score=accuracy_score(y[b],oof[b].argmax(1));stats.append({'fold':fold,'accuracy':score})
    if args.save_models:
        dest=root/(args.tag+'_models');dest.mkdir(exist_ok=True)
        torch.save(m.state_dict(),dest/f'fold{fold}_e{args.epochs}_s{args.seed}.pt')
    print('FOLD',fold,score,flush=True)
    del m,opt;schedule=None;gc.collect();torch.cuda.empty_cache()
name=f'{args.tag}_e{args.epochs}_s{args.seed}'
np.savez_compressed(root/(name+'.npz'),oof=oof,test=test,classes=enc.classes_)
report={'model':model_id,'accuracy':accuracy_score(y,oof.argmax(1)),'macro_f1':f1_score(y,oof.argmax(1),average='macro'),'folds':stats,'seconds':time.time()-start,'args':vars(args),
        'validation':f'5-fold grouped by unique primary symptom. Fixed {args.epochs}-epoch training declared before evaluating; full/core text augmentation; no handwritten fault-classification rules.'}
(root/(name+'.json')).write_text(json.dumps(report,indent=2));print('TOTAL',report,flush=True)
