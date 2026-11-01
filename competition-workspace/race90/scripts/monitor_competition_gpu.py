"""Read-only monitoring of the already-authorized competition Pod."""
from pathlib import Path
import csv,io,json,shlex,subprocess,time
from datetime import datetime,timezone

root=Path(__file__).resolve().parents[1]
live=root/'live';out=live/'gpu-monitor.jsonl'
remote_code=r'''
import subprocess,json,time,shutil
from pathlib import Path
def command(args):
 p=subprocess.run(args,capture_output=True,text=True,timeout=10)
 return p.stdout.strip() if p.returncode==0 else {'error':p.stderr.strip(),'returncode':p.returncode}
logs={}
for folder in ['/workspace/competition-radar','/workspace/competition-vision','/workspace/competition-text']:
 for p in sorted(Path(folder).glob('*.log'),key=lambda p:p.stat().st_mtime,reverse=True)[:4]:
  lines=p.read_text(errors='replace').replace('\r','\n').splitlines()
  if lines:logs[p.name]={'modified':p.stat().st_mtime,'last':lines[-1][-700:]}
processes=command(['ps','-eo','pid,etime,pcpu,pmem,args'])
if isinstance(processes,str):processes=[s for s in processes.splitlines() if '/bin/python' in s and any(n in s for n in ['radar_','vision_','text_']) and ' -c ' not in s]
disk=shutil.disk_usage('/workspace')
print(json.dumps({'observed_at':time.time(),'gpus_csv':command(['nvidia-smi','--query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total,power.draw','--format=csv,noheader,nounits']),'compute_processes_csv':command(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv,noheader,nounits']),'processes':processes,'logs':logs,'disk_used_bytes':disk.used,'disk_total_bytes':disk.total}))
'''
remote_python='/workspace/race90-27edfe8858d7/.venv/bin/python'
ssh=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','-i','/config/.ssh/runpod_codex_ed25519','-p','14949','root@154.54.102.25',remote_python+' -c '+shlex.quote(remote_code)]
idle_streak={}
print('Monitoring existing Pod vvpxznt3e1lvyn every 20 seconds until accuracy deadline.',flush=True)
while True:
    schedule=json.loads((live/'competition-schedule.json').read_text())
    if time.time()>=schedule['accuracy_deadline']:break
    try:
        result=subprocess.run(ssh,capture_output=True,text=True,timeout=25)
        if result.returncode:raise RuntimeError(result.stderr[-500:])
        row=json.loads(result.stdout);row['pod_id']='vvpxznt3e1lvyn';row['warnings']=[]
        gpus=[]
        for values in csv.reader(io.StringIO(row['gpus_csv'])):
            v=[x.strip() for x in values]
            gpu={'index':int(v[0]),'uuid':v[1],'name':v[2],'utilization_percent':int(v[3]),'memory_used_mib':int(v[4]),'memory_total_mib':int(v[5]),'power_w':float(v[6])}
            gpus.append(gpu);idx=gpu['index'];idle_streak[idx]=(idle_streak.get(idx,0)+1) if gpu['utilization_percent']<10 else 0
            if idle_streak[idx]>=3:row['warnings'].append(f'GPU {idx}: low utilization for {idle_streak[idx]} samples; inspect active task/queue.')
            if gpu['memory_used_mib']/gpu['memory_total_mib']>.92:row['warnings'].append(f'GPU {idx}: memory above 92%.')
        row['gpus']=gpus
        if row['disk_used_bytes']/row['disk_total_bytes']>.9:row['warnings'].append('Workspace disk usage above 90%.')
    except Exception as exc:row={'observed_at':time.time(),'error':str(exc)}
    row['observed_utc']=datetime.fromtimestamp(row['observed_at'],timezone.utc).isoformat()
    with out.open('a') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    temp=live/'gpu-monitor-latest.tmp';temp.write_text(json.dumps(row,ensure_ascii=False,indent=2));temp.replace(live/'gpu-monitor-latest.json')
    print(row['observed_utc'],[(g['index'],g['utilization_percent'],g['memory_used_mib']) for g in row.get('gpus',[])],row.get('warnings',row.get('error')),flush=True)
    time.sleep(20)
print('Accuracy phase ended; monitoring stopped.',flush=True)
