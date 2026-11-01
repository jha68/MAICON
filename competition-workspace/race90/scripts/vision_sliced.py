"""High-resolution overlapping crop detections for independent count review."""
from pathlib import Path
import json,time
import numpy as np
from PIL import Image
from ultralytics import YOLO

root=Path('/workspace/competition-vision')
model=YOLO(root/'yolov8x.pt')
clip={r['filename']:r['category'] for r in json.loads((root/'clip_predictions.json').read_text())}
out={};start=time.time()
for path in sorted((root/'data/aerial_objects').glob('*.jpg')):
    if clip[path.name]!='airplane':continue
    # Small partial aircraft benefit most; restrict output to airplane detections.
    im=Image.open(path).convert('RGB');w,h=im.size
    boxes=[]
    for scale in [.5,.75,1.0]:
        cw,ch=int(w*scale),int(h*scale)
        xs=sorted(set([0,max(0,(w-cw)//2),max(0,w-cw)]))
        ys=sorted(set([0,max(0,(h-ch)//2),max(0,h-ch)]))
        for x in xs:
            for y in ys:
                crop=im.crop((x,y,x+cw,y+ch))
                result=model.predict(crop,imgsz=1280,conf=.12,iou=.6,classes=[4],device=1,verbose=False)[0]
                for b in result.boxes:
                    coords=b.xyxy[0].cpu().numpy();coords[[0,2]]+=x;coords[[1,3]]+=y
                    boxes.append({'box':coords.tolist(),'conf':float(b.conf[0]),'crop':[x,y,cw,ch]})
    out[path.name]=boxes
    print(path.name,len(boxes),round(time.time()-start,1),flush=True)
    (root/'sliced_airplane_predictions.json').write_text(json.dumps(out))
