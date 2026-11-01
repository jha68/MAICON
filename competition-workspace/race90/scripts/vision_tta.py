from pathlib import Path
import json
import numpy as np
from PIL import Image
from ultralytics import YOLO
root=Path('/workspace/competition-vision');model=YOLO('yolov8s.pt');out=[]
files=sorted((root/'data/aerial_objects').glob('*.jpg'))
for i,p in enumerate(files):
 im=np.array(Image.open(p).convert('RGB'))[:,:,::-1];h,w=im.shape[:2];views=[]
 for k in [0,1,2,3]:
  r=model.predict(np.ascontiguousarray(np.rot90(im,k)),imgsz=640,device=2,conf=.05,iou=.35,classes=[4,14],verbose=False)[0]
  boxes=r.boxes.xyxy.cpu().numpy()
  converted=[]
  for x1,y1,x2,y2 in boxes:
   if k==0:b=[x1,y1,x2,y2]
   elif k==1:b=[w-y2,x1,w-y1,x2]
   elif k==2:b=[w-x2,h-y2,w-x1,h-y1]
   else:b=[y1,h-x2,y2,h-x1]
   converted.append([float(v) for v in b])
  views.append({'rotation':k,'boxes':converted,'conf':r.boxes.conf.cpu().tolist(),'cls':r.boxes.cls.cpu().tolist()})
 out.append({'filename':p.name,'shape':[h,w],'views':views})
 if i%30==0:print(i,flush=True)
(root/'tta_base_predictions.json').write_text(json.dumps(out))
