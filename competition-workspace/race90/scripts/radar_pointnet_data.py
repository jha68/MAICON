from pathlib import Path
import numpy as np,pandas as pd
from custom_adapters.radar_improved import sensor_geometry

root=Path('competition/q1');train=pd.read_csv(root/'data/train.csv');geo=sensor_geometry(train)
for split in ['train','test']:
    d=pd.read_csv(root/'data'/f'{split}.csv');clouds=[];ids=[]
    for sid,g in d.groupby('sample_id',sort=True):
        t=g.t_ms.to_numpy()/1000;t-=t.mean();p=g[['x_m','y_m']].to_numpy();p-=p.mean(0)
        vel=(t[:,None]*p).sum(0)/(np.dot(t,t)+1e-8);res=p-t[:,None]*vel
        u=np.array([np.array([np.cos(a),np.sin(a)])@geo[int(s)][:2] for a,s in zip(g.azimuth_rad,g.sensor_id)])
        r=g.rcs_dbsm.to_numpy();raw=g.radial_velocity_mps.to_numpy();comp=g.radial_velocity_comp_mps.to_numpy()
        features=np.c_[t/.25,p/np.array([5,2]),res/np.array([3,1]),r/10,raw/15,comp/10,(comp-raw)/10,u,np.eye(4)[g.sensor_id.to_numpy()-1]]
        clouds.append(features.astype('float32'));ids.append(sid)
    nmax=max(map(len,clouds));points=np.zeros((len(clouds),nmax,15),np.float32);mask=np.zeros((len(clouds),nmax),bool)
    for i,c in enumerate(clouds):points[i,:len(c)]=c;mask[i,:len(c)]=True
    np.savez_compressed(root/f'pointclouds_{split}.npz',points=points,mask=mask,ids=np.array(ids))
    print(split,points.shape,flush=True)
