"""Open-vocabulary audit of aircraft and partly visible aircraft tails."""
from pathlib import Path
import argparse,inspect,json,time
import torch
from PIL import Image
from transformers import AutoProcessor,AutoModelForZeroShotObjectDetection

p=argparse.ArgumentParser();p.add_argument('--device',default='cuda:2');args=p.parse_args();torch.set_num_threads(4)
root=Path('/workspace/competition-vision');model_id='IDEA-Research/grounding-dino-base'
processor=AutoProcessor.from_pretrained(model_id)
model=AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(args.device).eval()
post=processor.post_process_grounded_object_detection
threshold_key='threshold' if 'threshold' in inspect.signature(post).parameters else 'box_threshold'
category={r['filename']:r['category'] for r in json.loads((root/'clip_predictions.json').read_text())}
out={};start=time.time()
for path in sorted((root/'data/aerial_objects').glob('*.jpg')):
    if category[path.name]!='airplane':continue
    im=Image.open(path).convert('RGB');rows=[]
    # A whole-frame view and an upper/background crop provide distinct scales.
    for tag,box in [('whole',(0,0,im.width,im.height)),('background',(0,0,im.width,int(im.height*.7)))]:
        crop=im.crop(box)
        batch=processor(images=crop,text='an airplane. an aircraft tail.',return_tensors='pt').to(args.device)
        with torch.inference_mode():outputs=model(**batch)
        result=post(outputs,batch.input_ids,target_sizes=[crop.size[::-1]],text_threshold=.25,**{threshold_key:.22})[0]
        labels=result.get('text_labels',result.get('labels'))
        for coordinates,score,label in zip(result['boxes'].cpu().tolist(),result['scores'].cpu().tolist(),labels):
            coordinates[0]+=box[0];coordinates[2]+=box[0];coordinates[1]+=box[1];coordinates[3]+=box[1]
            rows.append({'box':coordinates,'score':score,'label':str(label),'view':tag})
    out[path.name]=rows
    if len(out)%10==0:print(len(out),'seconds',round(time.time()-start,1),flush=True)
    (root/'grounding_predictions.json').write_text(json.dumps(out))
print('DONE',len(out),'seconds',round(time.time()-start,1),flush=True)
