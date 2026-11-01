from pathlib import Path
import json
from ultralytics import YOLO
root=Path('/workspace/competition-vision')
for model_name,size in [('yolov8s.pt',960),('yolov8x.pt',1280)]:
 m=YOLO(model_name);out=[]
 for r in m.predict(source=str(root/'data/aerial_objects'),imgsz=size,device=2,conf=.03,iou=.6,classes=[4,14],stream=True,verbose=False):
  out.append({'filename':Path(r.path).name,'shape':list(r.orig_shape),'boxes':r.boxes.xyxy.cpu().tolist(),'conf':r.boxes.conf.cpu().tolist(),'cls':r.boxes.cls.cpu().tolist()})
 (root/(Path(model_name).stem+'_base_predictions.json')).write_text(json.dumps(out))
 print('DONE',model_name,flush=True)
