import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel

root=Path('/workspace/competition-text')
torch.set_num_threads(6)
name='intfloat/multilingual-e5-large'
tokenizer=AutoTokenizer.from_pretrained(name)
model=AutoModel.from_pretrained(name,torch_dtype=torch.float16).cuda().eval()
train=pd.read_csv(root/'train_core.csv');test=pd.read_csv(root/'test_core.csv')
texts=list(train.core)+list(test.core)+list(train.report)+list(test.report)
vectors=[]
with torch.inference_mode():
    for i in range(0,len(texts),32):
        batch=tokenizer(['query: '+t for t in texts[i:i+32]],padding=True,truncation=True,max_length=256,return_tensors='pt').to('cuda')
        h=model(**batch).last_hidden_state.float()
        mask=batch.attention_mask.unsqueeze(-1)
        e=(h*mask).sum(1)/mask.sum(1)
        vectors.append(torch.nn.functional.normalize(e,dim=1).cpu().numpy())
        print(i,len(texts),flush=True)
np.savez_compressed(root/'embeddings.npz',vectors=np.concatenate(vectors))
print('saved',flush=True)
