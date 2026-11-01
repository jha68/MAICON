from pathlib import Path
import numpy as np,pandas as pd
from custom_adapters.radar_improved import features,sensor_geometry

def robust_features(data):
 rows=[]
 for sid,g in data.groupby('sample_id',sort=True):
  f={'sample_id':sid};t=g.t_ms.to_numpy()/1000;t-=np.median(t)
  p=g[['x_m','y_m']].to_numpy();p-=np.median(p,axis=0)
  a=np.c_[np.ones(len(t)),t];weights=np.ones(len(t))
  for iteration in range(8):
   coef=np.linalg.lstsq(a*weights[:,None]**.5,p*weights[:,None]**.5,rcond=None)[0]
   res=p-a@coef;distance=np.linalg.norm(res,axis=1)
   scale=np.median(distance)+1e-4;weights=np.minimum(1,1.5*scale/(distance+1e-5))
  f['robust_track_speed']=np.linalg.norm(coef[1])
  for trim in [.6,.8,.9,1.0]:
   good=distance<=np.quantile(distance,trim);r=res[good];r=r-r.mean(0)
   cov=r.T@r/max(len(r)-1,1);eig=np.linalg.eigvalsh(cov)
   f['robust_eig0_'+str(trim)],f['robust_eig1_'+str(trim)]=eig
   f['robust_xstd_'+str(trim)]=np.sqrt(cov[0,0]);f['robust_ystd_'+str(trim)]=np.sqrt(cov[1,1])
   f['robust_shape_ratio_'+str(trim)]=eig[1]/max(eig[0],1e-6)
  for name,values in [('absvel',np.abs(g.radial_velocity_comp_mps.values)),('compvel',g.radial_velocity_comp_mps.values),('rcs',g.rcs_dbsm.values),('rawabsvel',np.abs(g.radial_velocity_mps.values))]:
   for q in [.05,.25,.5,.75,.95]:f[name+'_quantile_'+str(q)]=np.quantile(values,q)
   lo,hi=np.quantile(values,[.1,.9]);v=values[(values>=lo)&(values<=hi)]
   f[name+'_trimmean']=v.mean() if len(v) else values.mean();f[name+'_trimstd']=v.std() if len(v) else values.std()
  f['robust_outlier_ratio']=(distance>3*scale).mean()
  rows.append(f)
 return pd.DataFrame(rows).set_index('sample_id').fillna(0)

if __name__=='__main__':
 root=Path('competition/q1')
 for name,dest in [('train','features_robust.csv'),('test','features_test_robust.csv')]:
  data=pd.read_csv(root/'data'/f'{name}.csv')
  old=pd.read_csv(root/('features_improved.csv' if name=='train' else 'features_test_improved.csv'),index_col=0)
  x=pd.concat([old,robust_features(data)],axis=1).replace([np.inf,-np.inf],np.nan).fillna(0)
  x.to_csv(root/dest);print(name,x.shape,flush=True)
