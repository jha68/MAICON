"""Point-cloud topology and multimodality features, beyond summary moments."""
from pathlib import Path
import numpy as np,pandas as pd
from scipy.spatial.distance import pdist,squareform
from scipy.sparse.csgraph import minimum_spanning_tree
from sklearn.cluster import DBSCAN

def multimodal(x):
    x=x-x.mean(0);u,s,v=np.linalg.svd(x,full_matrices=False);projection=x@v[0]
    base=np.sum(x*x)+1e-6;out={}
    for k in [2,3]:
        centers=x[np.argsort(projection)[np.linspace(0,len(x)-1,k+2,dtype=int)[1:-1]]]
        for _ in range(12):
            dist=np.sum((x[:,None]-centers[None])**2,axis=-1);labels=dist.argmin(1)
            centers=np.array([x[labels==j].mean(0) if np.any(labels==j) else centers[j] for j in range(k)])
        sizes=np.bincount(labels,minlength=k)/len(x);sep=pdist(centers)
        out.update({f'k{k}_gain':1-np.sum((x-centers[labels])**2)/base,
                    f'k{k}_minfrac':sizes.min(),f'k{k}_maxfrac':sizes.max(),
                    f'k{k}_separation':sep.mean(),f'k{k}_separation_max':sep.max()})
    return out

def compute(data):
    rows=[]
    for sid,g in data.groupby('sample_id',sort=True):
        if len(g)>128:g=g.iloc[np.linspace(0,len(g)-1,128,dtype=int)]
        n=len(g);f={'sample_id':sid};xy=g[['x_m','y_m']].to_numpy();xy-=xy.mean(0)
        t=g.t_ms.to_numpy()/1000;t-=t.mean();a=np.c_[np.ones(n),t]
        inv=pd.factorize(g.sensor_id)[0];sensor=np.eye(inv.max()+1)[inv];aa=np.c_[sensor,t]
        for name,z in [('position',xy),('motion',xy-a@np.linalg.lstsq(a,xy,rcond=None)[0]),('sensor',xy-aa@np.linalg.lstsq(aa,xy,rcond=None)[0])]:
            distances=pdist(z);matrix=squareform(distances);near=matrix.copy();np.fill_diagonal(near,np.inf)
            edges=minimum_spanning_tree(matrix+np.eye(n)*1e-10).data
            for label,values in [('pair',distances),('nearest',near.min(1)),('tree',edges)]:
                if len(values)==0:values=np.array([0.])
                for q in [.1,.5,.9,1.]:f[f'cluster_{name}_{label}_q{q}']=np.quantile(values,q)
                f[f'cluster_{name}_{label}_mean']=values.mean()
            for eps in [.2,.5,1.,2.]:
                labels=DBSCAN(eps=eps,min_samples=2).fit_predict(z)
                counts=np.bincount(labels[labels>=0]) if np.any(labels>=0) else np.array([0])
                f[f'cluster_{name}_eps{eps}_clusters']=len(set(labels))-(int(-1 in labels))
                f[f'cluster_{name}_eps{eps}_largest']=counts.max()/n
                f[f'cluster_{name}_eps{eps}_noise']=np.mean(labels<0)
            f.update({f'cluster_{name}_{k}':val for k,val in multimodal(z).items()})
        rcs=g.rcs_dbsm.to_numpy();vel=g.radial_velocity_comp_mps.to_numpy()
        z=xy-aa@np.linalg.lstsq(aa,xy,rcond=None)[0]
        for name,cloud in [('rcsv',np.c_[rcs/10,vel/3]),('spacetime',np.c_[z,rcs/10,vel/3])]:
            f.update({f'cluster_{name}_{k}':val for k,val in multimodal(cloud).items()})
        rows.append(f)
    return pd.DataFrame(rows).set_index('sample_id').replace([np.inf,-np.inf],np.nan).fillna(0)

if __name__=='__main__':
    root=Path('competition/q1')
    for split in ['train','test']:
        x=compute(pd.read_csv(root/'data'/f'{split}.csv'));x.to_csv(root/f'cluster_features_{split}.csv');print(split,x.shape,flush=True)
