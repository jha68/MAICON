from pathlib import Path
import argparse,shutil,random,json
from ultralytics import YOLO

parser=argparse.ArgumentParser();parser.add_argument('--model',default='yolov8s.pt');parser.add_argument('--name',default='drone_s');parser.add_argument('--device',default='0');parser.add_argument('--epochs',type=int,default=120);parser.add_argument('--imgsz',type=int,default=640);args=parser.parse_args()
root=Path('/workspace/competition-vision');src=root/'data/transfer_learning';ds=root/'dataset'
images=sorted([p for p in src.iterdir() if p.suffix.lower() in {'.jpg','.jpeg','.png'}])
rng=random.Random(42);order=list(range(len(images)));rng.shuffle(order);val=set(order[:10])
for i,p in enumerate(images):
 split='val' if i in val else 'train'
 for sub in ['images','labels']:(ds/sub/split).mkdir(parents=True,exist_ok=True)
 shutil.copy2(p,ds/'images'/split/p.name);shutil.copy2(p.with_suffix('.txt'),ds/'labels'/split/(p.stem+'.txt'))
(ds/'data.yaml').write_text('path: '+str(ds)+'\ntrain: images/train\nval: images/val\nnames:\n  0: drone\n')
model=YOLO(args.model)
model.train(data=str(ds/'data.yaml'),epochs=args.epochs,imgsz=args.imgsz,batch=16,device=args.device,workers=6,
 project=str(root/'runs'),name=args.name,exist_ok=True,seed=42,patience=60,cache=True,
 close_mosaic=15,optimizer='AdamW',lr0=.0003,lrf=.05,weight_decay=.0005,
 warmup_bias_lr=0.0,warmup_epochs=5,nbs=16,amp=False,
 degrees=10,translate=.1,scale=.4,fliplr=.5,mosaic=.6,mixup=0,plots=True)
best=root/'runs'/args.name/'weights/best.pt'
m=YOLO(str(best));out=[]
for r in m.predict(source=str(root/'data/aerial_objects'),imgsz=args.imgsz,device=args.device,conf=.05,iou=.55,stream=True,verbose=False):
 out.append({'filename':Path(r.path).name,'shape':list(r.orig_shape),'boxes':r.boxes.xyxy.cpu().tolist(),'conf':r.boxes.conf.cpu().tolist(),'cls':r.boxes.cls.cpu().tolist()})
(root/(args.name+'_predictions.json')).write_text(json.dumps(out))
print('DONE',args.name,flush=True)
