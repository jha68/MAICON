"""Short local YOLO transfer-learning run on two CPU cores, without saved predictions."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[k]='2'
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['YOLO_AUTOINSTALL']='false'
os.environ['WANDB_DISABLED']='true'
if hasattr(os,'sched_getaffinity'):
    cpus=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,set(cpus[2:4] if len(cpus)>=4 else cpus[:2]))
from pathlib import Path
import time,json,random,shutil,gc,resource
import numpy as np,pandas as pd,torch
torch.set_num_threads(2)
torch.set_num_interop_threads(1)
from ultralytics import YOLO

root=Path('/workspace/competition-vision');out=root/'cpu_training_check';out.mkdir(exist_ok=True)
src=root/'data/transfer_learning';dataset=out/'dataset'
images=sorted(p for p in src.iterdir() if p.suffix.lower() in {'.jpg','.jpeg','.png'})
order=list(range(len(images)));random.Random(42).shuffle(order);valid=set(order[:10])
for i,p in enumerate(images):
    split='val' if i in valid else 'train'
    for kind in ('images','labels'):(dataset/kind/split).mkdir(parents=True,exist_ok=True)
    shutil.copy2(p,dataset/'images'/split/p.name)
    shutil.copy2(p.with_suffix('.txt'),dataset/'labels'/split/(p.stem+'.txt'))
(dataset/'data.yaml').write_text('path: '+str(dataset)+'\ntrain: images/train\nval: images/val\nnames:\n  0: drone\n')
truth={p.name:len([s for s in p.with_suffix('.txt').read_text().splitlines() if s.strip()]) for i,p in enumerate(images) if i in valid}
report={'torch':torch.__version__,'cpu_affinity':sorted(os.sched_getaffinity(0)),'validation_images':truth,'validation_limitation':'Only 10 held-out drone images. No airplane/bird or hidden-test ground truth. Earlier best checkpoints were selected using this same validation set.','runs':[]}

def evaluate(path,name,elapsed=None):
    start=time.perf_counter();m=YOLO(str(path));counts={}
    for r in m.predict(source=str(dataset/'images/val'),device='cpu',imgsz=640,conf=.3,iou=.55,stream=True,verbose=False):
        counts[Path(r.path).name]=int((r.boxes.cls==0).sum().item())
    metric=float(np.mean([counts[n]==count for n,count in truth.items()]))
    item={'name':name,'heldout_drone_count_accuracy':metric,'predicted_counts':counts,'training_seconds':elapsed,'validation_seconds':time.perf_counter()-start,'max_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    report['runs'].append(item);(out/'report.json').write_text(json.dumps(report,indent=2));print('COUNT_RESULT',json.dumps(item),flush=True)
    del m;gc.collect()

for name in ('drone_s_stable','drone_m_stable'):
    try:evaluate(root/'runs'/name/'weights/best.pt',name)
    except Exception as exc:print('BASELINE_LOAD_FAILED',name,type(exc).__name__,str(exc),flush=True)

start=time.perf_counter()
def checkpoint(trainer):
    epoch=trainer.epoch+1
    if epoch in (5,10,20):
        evaluate(trainer.last,'cpu_yolov8s_epoch_'+str(epoch),time.perf_counter()-start)

model=YOLO(str(root/'yolov8s.pt'))
model.add_callback('on_pretrain_routine_start',lambda trainer:torch.set_num_threads(2))
model.add_callback('on_model_save',checkpoint)
model.train(data=str(dataset/'data.yaml'),epochs=20,imgsz=384,batch=2,device='cpu',workers=0,
            project=str(out/'runs'),name='drone_s_cpu',exist_ok=True,seed=42,patience=30,cache=False,
            close_mosaic=0,optimizer='AdamW',lr0=.001,lrf=.1,weight_decay=.0005,
            warmup_bias_lr=0.0,warmup_epochs=1,nbs=2,amp=False,
            degrees=10,translate=.1,scale=.4,fliplr=.5,mosaic=0,mixup=0,plots=False,verbose=False)
report['complete_training_seconds']=time.perf_counter()-start
report['max_rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
(out/'report.json').write_text(json.dumps(report,indent=2))
print('CPU_VISION_COMPARISON_DONE',json.dumps(report),flush=True)
