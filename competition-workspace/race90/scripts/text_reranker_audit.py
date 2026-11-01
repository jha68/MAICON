"""Cross-encoder matching to labeled symptom prototypes, without classification rules."""
from pathlib import Path
import argparse,json,time
import numpy as np,pandas as pd,torch
from transformers import AutoTokenizer,AutoModelForSequenceClassification
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score,f1_score

p=argparse.ArgumentParser();p.add_argument('--device',default='cuda:2');args=p.parse_args()
torch.set_num_threads(4);root=Path('/workspace/competition-text')
train=pd.read_csv(root/'train_core.csv');test=pd.read_csv(root/'test_core.csv')
prototypes=train.drop_duplicates('core').reset_index(drop=True)
queries=pd.concat([prototypes[['core']],test[['core']]]).drop_duplicates('core').reset_index(drop=True)
model_id='BAAI/bge-reranker-v2-m3';tokenizer=AutoTokenizer.from_pretrained(model_id)
model=AutoModelForSequenceClassification.from_pretrained(model_id,torch_dtype=torch.float16).to(args.device).eval()
pairs=[(q,c) for q in queries.core for c in prototypes.core];scores=[];start=time.time()
for i in range(0,len(pairs),64):
    batch=tokenizer(pairs[i:i+64],padding=True,truncation=True,max_length=256,return_tensors='pt').to(args.device)
    with torch.inference_mode():scores.extend(model(**batch).logits.flatten().float().cpu().tolist())
    if i%1024==0:print(i,len(pairs),'seconds',round(time.time()-start,1),flush=True)
matrix=np.asarray(scores).reshape(len(queries),len(prototypes))
np.savez_compressed(root/'reranker_scores.npz',scores=matrix,queries=queries.core.to_numpy(str),prototypes=prototypes.core.to_numpy(str),labels=prototypes.fault_type.to_numpy(str))
qid={s:i for i,s in enumerate(queries.core)};pid={s:i for i,s in enumerate(prototypes.core)}
prediction=np.empty(len(train),dtype=object)
for a,b in StratifiedGroupKFold(5,shuffle=True,random_state=42).split(train,train.fault_type,train.core):
    columns=np.array(sorted({pid[s] for s in train.iloc[a].core}))
    rows=np.array([qid[s] for s in train.iloc[b].core]);nearest=columns[matrix[np.ix_(rows,columns)].argmax(1)]
    prediction[b]=prototypes.iloc[nearest].fault_type.to_numpy()
te_rows=np.array([qid[s] for s in test.core]);nearest=matrix[te_rows].argmax(1)
test_prediction=prototypes.iloc[nearest].fault_type.to_numpy()
test_out=test[['id','report','core']].copy();test_out['prediction']=test_prediction
test_out['nearest_core']=prototypes.iloc[nearest].core.to_numpy();test_out['score']=matrix[te_rows,nearest]
test_out.to_csv(root/'reranker_test_predictions.csv',index=False)
report={'model':model_id,'method':'Cross-encoder ranks supplied labeled symptom prototypes; no handwritten fault classification rules. Five-fold grouped validation excludes all copies of held-out cores.',
        'grouped_accuracy':accuracy_score(train.fault_type,prediction),'grouped_macro_f1':f1_score(train.fault_type,prediction,average='macro'),
        'train_errors':[{'id':train.iloc[i].id,'core':train.iloc[i].core,'actual':train.iloc[i].fault_type,'prediction':prediction[i]} for i in range(len(train)) if prediction[i]!=train.iloc[i].fault_type],
        'seconds':time.time()-start}
(root/'reranker_validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
print('DONE',report['grouped_accuracy'],report['grouped_macro_f1'],flush=True)
