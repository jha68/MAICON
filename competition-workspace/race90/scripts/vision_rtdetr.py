"""Independent Transformer detector audit; outputs are candidates for visual review."""
from pathlib import Path
import argparse, json, time
import torch
from ultralytics import RTDETR

p=argparse.ArgumentParser();p.add_argument('--device',type=int,default=2);args=p.parse_args()
torch.set_num_threads(4)
root=Path('/workspace/competition-vision')
category={r['filename']:r['category'] for r in json.loads((root/'clip_predictions.json').read_text())}
model=RTDETR(root/'rtdetr-x.pt')
for size in [640,1280]:
    out=[];start=time.time()
    files=[str(p) for p in sorted((root/'data/aerial_objects').glob('*.jpg')) if category[p.name]!='drone']
    # A Python source list is loaded as one in-memory batch by Ultralytics.
    # Use one filename per call to keep inference memory bounded.
    for i,filename in enumerate(files):
        result=model.predict(source=filename,imgsz=size,device=args.device,conf=.08,
                             classes=[4,14],batch=1,verbose=False)[0]
        out.append({'filename':Path(result.path).name,'shape':list(result.orig_shape),
                    'boxes':result.boxes.xyxy.cpu().tolist(),'conf':result.boxes.conf.cpu().tolist(),
                    'cls':result.boxes.cls.cpu().tolist()})
        if i%20==19:print(size,i+1,'seconds',round(time.time()-start,1),flush=True)
    (root/f'rtdetr_{size}_predictions.json').write_text(json.dumps(out))
    print('DONE',size,len(out),'seconds',round(time.time()-start,1),flush=True)
