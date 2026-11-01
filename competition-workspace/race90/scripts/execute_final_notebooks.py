"""Execute delivered notebooks with CPU affinity and an address-space ceiling."""
from pathlib import Path
import argparse,hashlib,json,os,platform,resource,time
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='2'
os.environ['CUDA_VISIBLE_DEVICES']=''
p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--kernel',default='python3');p.add_argument('--limit-memory',action='store_true');args=p.parse_args()
if hasattr(os,'sched_getaffinity'):
    affinity=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,set(affinity[:2]))
if args.limit_memory:resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3))
import nbformat
from nbclient import NotebookClient
import numpy,pandas
root=Path(args.root).resolve();report={'python':platform.python_version(),'architecture':platform.machine(),'numpy':numpy.__version__,'pandas':pandas.__version__,'cpu_affinity':sorted(os.sched_getaffinity(0)),'cuda_visible_devices':os.environ['CUDA_VISIBLE_DEVICES'],'address_space_limit_bytes':4*1024**3 if args.limit_memory else None,'online_kernel_directly_verified':False,'notebooks':[]}
for folder in sorted(root.glob('Problem #*')):
    path=next(folder.glob('*.ipynb'));nb=nbformat.read(path,as_version=4);start=time.monotonic()
    client=NotebookClient(nb,timeout=120,kernel_name=args.kernel,resources={'metadata':{'path':str(folder)}})
    client.execute();nbformat.validate(nb);nbformat.write(nb,path)
    errors=[o for c in nb.cells if c.cell_type=='code' for o in c.get('outputs',[]) if o.output_type=='error']
    assert not errors
    csv_path=folder/'submission.csv';frame=pandas.read_csv(csv_path)
    item={'question':int(folder.name.split('#')[-1]),'notebook':path.name,'executed_code_cells':sum(c.cell_type=='code' and c.execution_count is not None for c in nb.cells),'seconds':round(time.monotonic()-start,3),'rows':len(frame),'columns':frame.columns.tolist(),'csv_sha256':hashlib.sha256(csv_path.read_bytes()).hexdigest(),'max_child_rss_kib':resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss}
    report['notebooks'].append(item);print(json.dumps(item,ensure_ascii=False),flush=True)
    (root/'execution-report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
print('ALL_NOTEBOOKS_PASSED',flush=True)
