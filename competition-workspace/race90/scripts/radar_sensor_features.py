"""Separate sensor-dependent offsets from object shape and motion."""
from pathlib import Path
import numpy as np,pandas as pd
from scipy.spatial.distance import pdist
from scipy.stats import skew,kurtosis
from custom_adapters.radar_improved import sensor_geometry

def summarize(f,name,v):
    v=np.asarray(v);f[name+'_std']=np.std(v)
    f[name+'_span80']=np.quantile(v,.9)-np.quantile(v,.1)
    f[name+'_span50']=np.quantile(v,.75)-np.quantile(v,.25)
    f[name+'_mad']=np.median(np.abs(v-np.median(v)))

def compute(data,geo):
    rows=[]
    for sid,g in data.groupby('sample_id',sort=True):
        n=len(g);f={'sample_id':sid}
        t=g.t_ms.to_numpy()/1000;t-=t.mean();xy=g[['x_m','y_m']].to_numpy()
        sens=g.sensor_id.to_numpy();unique,inv=np.unique(sens,return_inverse=True)
        z=np.eye(len(unique))[inv];a=np.c_[z,t]
        positions=[]
        for mode in ['ols','huber']:
            weights=np.ones(n)
            for it in range(1 if mode=='ols' else 8):
                coef=np.linalg.lstsq(a*weights[:,None]**.5,xy*weights[:,None]**.5,rcond=None)[0]
                res=xy-a@coef;distance=np.linalg.norm(res,axis=1);scale=np.median(distance)+1e-5
                weights=np.minimum(1,1.5*scale/(distance+1e-5))
            name='sensorfixed_'+mode
            f[name+'_speed']=np.linalg.norm(coef[-1]);f[name+'_vx'],f[name+'_vy']=coef[-1]
            f[name+'_dof']=max(n-len(unique)-1,0)
            intercepts=coef[:-1]
            f[name+'_offset_spread']=np.mean(pdist(intercepts)) if len(intercepts)>1 else 0
            for trim in [1,.8]:
                good=distance<=np.quantile(distance,trim);r=res[good];r=r-r.mean(0)
                cov=r.T@r/max(len(r)-len(unique)-1,1);eig,rot=np.linalg.eigh(cov)
                pca=r@rot
                suffix=f'{name}_trim{trim}'
                f[suffix+'_eig0'],f[suffix+'_eig1']=eig
                f[suffix+'_xvar'],f[suffix+'_yvar']=np.diag(cov)
                f[suffix+'_area']=np.sqrt(max(np.linalg.det(cov),0))
                for j in range(2):summarize(f,suffix+'_pca'+str(j),pca[:,j])
            if mode=='ols':positions=res
        # Removing each sensor's own mean avoids cross-sensor RCS and velocity offsets.
        for col in ['rcs_dbsm','radial_velocity_mps','radial_velocity_comp_mps']:
            v=g[col].to_numpy();coef=np.linalg.lstsq(a,v,rcond=None)[0];res=v-a@coef
            name='sensorfixed_'+col
            summarize(f,name,res);f[name+'_slope']=coef[-1]
            within=v-z@np.linalg.lstsq(z,v,rcond=None)[0]
            f[name+'_within_std']=np.std(within)
            f[name+'_mean_spread']=np.ptp(coef[:-1])
        # Approximate support extent using the brightest returns as well as all returns.
        rcs=g.rcs_dbsm.to_numpy();rr=g.range_m.to_numpy()
        for temp in [3,10]:
            w=np.exp(np.clip((rcs-rcs.max())/temp,-25,0));w/=w.sum()
            cen=(positions*w[:,None]).sum(0);r=positions-cen
            cov=(r*w[:,None]).T@r;ev=np.linalg.eigvalsh(cov)
            f[f'rcsweighted_{temp}_eig0'],f[f'rcsweighted_{temp}_eig1']=ev
            f[f'rcsweighted_{temp}_effective_n']=1/np.sum(w*w)
        # Normalize angular extent by range; retain radial extent in meters.
        directions=np.array([np.array([np.cos(v),np.sin(v)])@geo[int(s)][:2] for v,s in zip(g.azimuth_rad,sens)])
        radial=(positions*directions).sum(1);tangential=positions[:,0]*directions[:,1]-positions[:,1]*directions[:,0]
        summarize(f,'sensorfixed_radial',radial);summarize(f,'sensorfixed_tangential',tangential)
        summarize(f,'sensorfixed_angular',tangential/np.maximum(rr,1))
        f['sensorfixed_radial_tangent_ratio']=np.std(radial)/(np.std(tangential)+1e-5)
        # Temporal curvature cannot be explained by static sensor offsets.
        if n>len(unique)+2:
            aa=np.c_[a,t*t];c=np.linalg.lstsq(aa,xy,rcond=None)[0];res=xy-aa@c
            f['sensorfixed_acceleration']=np.linalg.norm(c[-1])*2
            f['sensorfixed_curve_resid']=np.mean(res*res)
        else:f['sensorfixed_acceleration']=f['sensorfixed_curve_resid']=0
        rows.append(f)
    return pd.DataFrame(rows).set_index('sample_id').replace([np.inf,-np.inf],np.nan).fillna(0)

if __name__=='__main__':
    root=Path('competition/q1');geo=sensor_geometry(pd.read_csv(root/'data/train.csv'))
    for split in ['train','test']:
        extra=compute(pd.read_csv(root/'data'/f'{split}.csv'),geo)
        extra.to_csv(root/f'sensor_features_{split}.csv')
        print(split,extra.shape,flush=True)
