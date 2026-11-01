from pathlib import Path
import json
from PIL import Image
import torch
from transformers import CLIPModel,CLIPProcessor
root=Path('/workspace/competition-vision');torch.set_num_threads(6)
name='openai/clip-vit-large-patch14'
m=CLIPModel.from_pretrained(name,torch_dtype=torch.float16).to('cuda:2').eval();p=CLIPProcessor.from_pretrained(name)
texts=['a photo of an airplane','a photo of a passenger aircraft','a photo of a military airplane',
       'a photo of a bird','a photo of birds','a photo of a peacock',
       'a photo of a quadcopter drone','a photo of a remote controlled drone','a photo of a toy helicopter']
with torch.inference_mode():
 tokens=p(text=texts,return_tensors='pt',padding=True).to('cuda:2');t=m.get_text_features(**tokens);t=t/t.norm(dim=1,keepdim=True)
 files=sorted((root/'data/aerial_objects').glob('*.jpg'));out=[]
 for i in range(0,len(files),16):
  ims=[Image.open(f).convert('RGB') for f in files[i:i+16]];batch=p(images=ims,return_tensors='pt').to('cuda:2');v=m.get_image_features(**batch);v=v/v.norm(dim=1,keepdim=True);scores=(v@t.T).float().cpu()
  for f,s in zip(files[i:i+16],scores):out.append({'filename':f.name,'scores':s.tolist(),'category':['airplane','bird','drone'][s.reshape(3,3).max(1).values.argmax().item()]})
  print(i,flush=True)
(root/'clip_predictions.json').write_text(json.dumps(out))
