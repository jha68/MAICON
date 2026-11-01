"""Additional radar features: infer ego motion from Doppler compensation."""
from pathlib import Path
import numpy as np
import pandas as pd
from custom_adapters.radar_improved import sensor_geometry


def robust_line(a, b, iterations=8):
    coef=np.linalg.lstsq(a,b,rcond=None)[0]
    for _ in range(iterations):
        r=b-a@coef
        dist=np.linalg.norm(r,axis=1) if r.ndim==2 else np.abs(r)
        scale=np.median(dist)+1e-6
        w=np.minimum(1.0,1.5*scale/(dist+1e-6))**0.5
        coef=np.linalg.lstsq(a*w[:,None],b*w[:,None] if b.ndim==2 else b*w,rcond=None)[0]
    return coef


def physics_features(data, geometry):
    rows=[]
    for sid,g in data.groupby('sample_id',sort=True):
        g=g.sort_values('t_ms');n=len(g)
        t=g.t_ms.to_numpy()/1000;t=t-t.mean()
        p=g[['x_m','y_m']].to_numpy();pc=p-p.mean(0)
        u=np.array([np.array([np.cos(a),np.sin(a)])@geometry[int(s)][:2] for a,s in zip(g.azimuth_rad,g.sensor_id)])
        mount=np.array([geometry[int(s)][2] for s in g.sensor_id])
        delta=(g.radial_velocity_comp_mps-g.radial_velocity_mps).to_numpy()
        raw=g.radial_velocity_mps.to_numpy();comp=g.radial_velocity_comp_mps.to_numpy()
        f={'sample_id':sid}
        ego=np.linalg.lstsq(u,delta,rcond=None)[0]
        yawcol=-mount[:,1]*u[:,0]+mount[:,0]*u[:,1]
        A=np.c_[u,yawcol]
        egofull=np.linalg.solve(A.T@A+np.diag([1e-5,1e-5,.001]),A.T@delta)
        for name, v in [('ego',ego),('ego_full',egofull[:2])]:
            f[name+'_vx'],f[name+'_vy']=v
            f[name+'_speed']=np.linalg.norm(v)
        f['ego_yaw']=egofull[2]
        f['ego_residual']=np.std(delta-u@ego)
        f['ego_full_residual']=np.std(delta-A@egofull)
        f['ego_forward_median']=np.median(delta/np.maximum(u[:,0],.05))
        f['ego_forward_std']=np.std(delta/np.maximum(u[:,0],.05))
        motion=np.c_[np.ones(n),t]
        for kind in ['ols','robust','frame']:
            if kind=='frame':
                fr=g.groupby('t_ms')[['x_m','y_m']].median();ft=fr.index.to_numpy()/1000;ft-=ft.mean()
                vel=np.linalg.lstsq(np.c_[np.ones(len(fr)),ft],fr.to_numpy(),rcond=None)[0][1]
            else:
                coef=robust_line(motion,p) if kind=='robust' else np.linalg.lstsq(motion,p,rcond=None)[0]
                vel=coef[1]
            for variant,correction in [('trans',ego),('yaw',egofull[:2]+egofull[2]*np.array([-p[:,1].mean(),p[:,0].mean()]))]:
                v=vel+correction
                f[kind+'_'+variant+'_vx'],f[kind+'_'+variant+'_vy']=v
                f[kind+'_'+variant+'_speed']=np.linalg.norm(v)
            res=pc-t[:,None]*vel
            absolute=vel+ego
            for dop_name,dop in [('raw',raw),('comp',comp)]:
                rr=dop-u@(vel if dop_name=='raw' else absolute)
                f[kind+'_'+dop_name+'_noise_std']=np.std(rr)
                f[kind+'_'+dop_name+'_noise_mad']=np.median(np.abs(rr-np.median(rr)))
                f[kind+'_'+dop_name+'_noise_bias']=np.median(rr)
            for basis,axis in [('movement',absolute),('view',u.mean(0))]:
                axis=axis/(np.linalg.norm(axis)+1e-8);normal=np.array([-axis[1],axis[0]])
                for direction,v in [('longitudinal',res@axis),('lateral',res@normal)]:
                    f[kind+'_'+basis+'_'+direction+'_std']=np.std(v)
                    f[kind+'_'+basis+'_'+direction+'_span']=np.quantile(v,.9)-np.quantile(v,.1)
                    f[kind+'_'+basis+'_'+direction+'_range']=np.ptp(v)
        # Fuse position and Doppler observations at several relative noise scales.
        for weight in [.01,.1,1,10,100]:
            v=np.linalg.solve(u.T@u+weight*np.dot(t,t)*np.eye(2),u.T@comp+weight*(t[:,None]*(pc+t[:,None]*ego)).sum(0))
            f['absolute_joint_speed_'+str(weight)]=np.linalg.norm(v)
            f['absolute_joint_noise_'+str(weight)]=np.std(comp-u@v)
        # Return statistics per sensor, then aggregate independent of sensor ID.
        vals=[]
        for sen,sg in g.groupby('sensor_id'):
            tt=sg.t_ms.to_numpy()/1000
            rv=sg.radial_velocity_comp_mps.to_numpy();rr=sg.rcs_dbsm.to_numpy()
            vals.append([len(sg),rv.mean(),rv.std(),np.quantile(rv,.9)-np.quantile(rv,.1),rr.mean(),rr.std(),sg.t_ms.nunique()])
        vals=np.array(vals)
        for j,name in enumerate(['n','vel_mean','vel_std','vel_span','rcs_mean','rcs_std','frames']):
            f['sensor_'+name+'_min']=vals[:,j].min();f['sensor_'+name+'_max']=vals[:,j].max()
            f['sensor_'+name+'_avg']=np.average(vals[:,j],weights=vals[:,0])
        frame_n=g.groupby('t_ms').size().to_numpy()
        for q in [.1,.5,.9]:f['per_frame_n_q'+str(q)]=np.quantile(frame_n,q)
        f['single_detection_fraction']=(frame_n==1).mean()
        f['max_frame_detections']=frame_n.max()
        f['duration']=np.ptp(t)
        f['timestamp_step_median']=np.median(np.diff(np.unique(t))) if len(np.unique(t))>1 else 0
        f['timestamp_step_std']=np.std(np.diff(np.unique(t))) if len(np.unique(t))>1 else 0
        rows.append(f)
    return pd.DataFrame(rows).set_index('sample_id').replace([np.inf,-np.inf],np.nan).fillna(0)


if __name__=='__main__':
    root=Path('competition/q1');train=pd.read_csv(root/'data/train.csv');geometry=sensor_geometry(train)
    for split in ['train','test']:
        data=pd.read_csv(root/'data'/f'{split}.csv')
        extra=physics_features(data,geometry)
        original=pd.read_csv(root/('features_robust.csv' if split=='train' else 'features_test_robust.csv'),index_col=0)
        extra.to_csv(root/f'physics_{split}.csv')
        pd.concat([original,extra],axis=1).to_csv(root/f'features_physics_{split}.csv')
        print(split,extra.shape,flush=True)
