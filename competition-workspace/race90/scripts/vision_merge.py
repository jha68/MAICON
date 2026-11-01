from pathlib import Path
import json
import numpy as np,pandas as pd
from PIL import Image,ImageDraw,ImageOps

root=Path('competition/q2')
def read(name):return {r['filename']:r for r in json.loads((root/name).read_text())}
clip=read('clip_predictions.json');drone_s=read('drone_s_stable_predictions.json');drone_m=read('drone_m_stable_predictions.json')
base_s=read('yolov8s_base_predictions.json');base_x=read('yolov8x_base_predictions.json')
tta=read('tta_base_predictions.json') if (root/'tta_base_predictions.json').exists() else {}
def overlap(a,b):
 x1,y1=np.maximum(a[:2],b[:2]);x2,y2=np.minimum(a[2:],b[2:]);inter=max(x2-x1,0)*max(y2-y1,0)
 area_a=(a[2]-a[0])*(a[3]-a[1]);area_b=(b[2]-b[0])*(b[3]-b[1])
 return inter/max(area_a+area_b-inter,1e-6),inter/max(min(area_a,area_b),1e-6)
def merge(boxes):
 boxes=sorted(boxes,key=lambda x:x[1],reverse=True);out=[]
 for box,conf,model in boxes:
  match=None
  for j,obj in enumerate(out):
   iou,iom=overlap(np.array(box),np.array(obj['box']))
   if iou>.3 or iom>.68:match=j;break
  if match is None:out.append({'box':box,'conf':conf,'models':[model]})
  else:
   obj=out[match];obj['models'].append(model)
   # Preserve the broader box when a model detected only one part.
   if conf>.25:obj['box']=[min(box[0],obj['box'][0]),min(box[1],obj['box'][1]),max(box[2],obj['box'][2]),max(box[3],obj['box'][3])]
 return out

records=[];details=[]
for fn in sorted(clip):
 cat=clip[fn]['category'];boxes=[]
 if cat=='drone':
  sources=[('ds',drone_s[fn]),('dm',drone_m[fn])];label=0
 else:
  label=4 if cat=='airplane' else 14
  sources=[('bs',base_s[fn]),('bx',base_x[fn])]
  if fn in tta:
   sources.extend(('r'+str(v['rotation']),v) for v in tta[fn]['views'] if cat=='bird' or v['rotation']==0)
 for name,r in sources:
  for box,conf,cl in zip(r['boxes'],r['conf'],r['cls']):
   if cl==label and conf>=.15:boxes.append((box,conf,name))
 merged=merge(boxes)
 accepted=[b for b in merged if b['conf']>=.55 or (b['conf']>=.3 and len(set(b['models']))>=2)]
 if not accepted and merged:accepted=[merged[0]]
 count=max(1,len(accepted))
 row={'filename':fn,'airplane':0,'bird':0,'drone':0};row[cat]=count;records.append(row)
 details.append({'filename':fn,'category':cat,'count':count,'boxes':accepted,'all_boxes':merged,'clip_scores':clip[fn]['scores']})
pd.DataFrame(records).to_csv(root/'submission.auto.csv',index=False)
(root/'merged_detections.json').write_text(json.dumps(details))
review=[d for d in details if d['count']!=1 or not d['boxes'] or max(b['conf'] for b in d['boxes'])<.55]
print('category counts',pd.Series([d['category'] for d in details]).value_counts().to_dict(),'review',len(review))
for d in review:print(d['filename'],d['category'],d['count'],[(round(b['conf'],2),b['models']) for b in d['boxes']])
dest=root/'review'
for page in range((len(review)+7)//8):
 canvas=Image.new('RGB',(1600,800),'white');draw=ImageDraw.Draw(canvas)
 for j,d in enumerate(review[page*8:page*8+8]):
  p=root/'data/aerial_objects'/d['filename'];im=Image.open(p).convert('RGB');idraw=ImageDraw.Draw(im)
  for k,b in enumerate(d['boxes']):idraw.rectangle(b['box'],outline='red',width=max(2,im.width//300));idraw.text((b['box'][0],b['box'][1]),str(k+1),fill='red',stroke_width=1)
  tile=ImageOps.contain(im,(390,350));x=(j%4)*400;y=(j//4)*400;canvas.paste(tile,(x,y));draw.text((x+5,y+360),d['filename']+' '+d['category']+' '+str(d['count']),fill='black')
 canvas.save(dest/f'detections_{page}.jpg')
