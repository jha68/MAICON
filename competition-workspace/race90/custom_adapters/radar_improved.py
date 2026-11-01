from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from custom_adapters.competition import radar_features


def sensor_geometry(data):
    transforms={}
    for sid,g in data.groupby('sensor_id'):
        a=np.c_[g.range_m*np.cos(g.azimuth_rad),g.range_m*np.sin(g.azimuth_rad),np.ones(len(g))]
        transforms[int(sid)]=np.linalg.lstsq(a,g[['x_m','y_m']].values,rcond=None)[0]
    return transforms


def features(data,geometry):
    basic=radar_features(data);rows=[]
    for sid,g in data.groupby('sample_id',sort=True):
        g=g.sort_values('t_ms');n=len(g);f={'sample_id':sid}
        t=g.t_ms.to_numpy()/1000;t=t-t.mean()
        pos=g[['x_m','y_m']].to_numpy();pos=pos-pos.mean(0)
        v=g.radial_velocity_comp_mps.to_numpy();r=g.rcs_dbsm.to_numpy()
        directions=[]
        for a,sen in zip(g.azimuth_rad,g.sensor_id):
            rot=geometry[int(sen)][:2,:]
            directions.append(np.array([np.cos(a),np.sin(a)])@rot)
        u=np.array(directions)
        velocity=np.linalg.lstsq(u,v,rcond=None)[0]
        f['doppler_speed']=np.linalg.norm(velocity)
        f['doppler_vx'],f['doppler_vy']=velocity
        f['doppler_resid_std']=np.std(v-u@velocity)
        f['doppler_resid_mad']=np.median(np.abs(v-u@velocity-np.median(v-u@velocity)))
        f['direction_condition']=np.linalg.cond(u.T@u)
        for ridge in [.01,.1,1,10]:
            vel=np.linalg.solve(u.T@u+ridge*np.eye(2),u.T@v)
            f['speed_ridge_'+str(ridge)]=np.linalg.norm(vel)
        # Joint least squares: positions at different times and radial velocities.
        for w in [.1,1,10]:
            vel=np.linalg.solve(u.T@u+w*np.dot(t,t)*np.eye(2),u.T@v+w*(t[:,None]*pos).sum(0))
            f['speed_joint_'+str(w)]=np.linalg.norm(vel)
        temporal_vel=(t[:,None]*pos).sum(0)/(np.dot(t,t)+1e-12)
        for mode,vel in [('raw',np.zeros(2)),('time',temporal_vel),('doppler',velocity)]:
            p=pos-t[:,None]*vel
            cov=p.T@p/max(n-1,1);e,rot=np.linalg.eigh(cov);pca=p@rot
            for i in range(2):
                f[mode+'_eig'+str(i)]=e[i]
                f[mode+'_pca_skew'+str(i)]=skew(pca[:,i])
                f[mode+'_pca_kurt'+str(i)]=kurtosis(pca[:,i])
                for q in [0.8,0.9,0.95,1.0]:
                    f[mode+'_span'+str(i)+'_'+str(q)]=np.quantile(pca[:,i],q)-np.quantile(pca[:,i],1-q)
            f[mode+'_xy_cov']=cov[0,1]
            f[mode+'_area']=np.sqrt(max(np.linalg.det(cov),0))
            f[mode+'_ratio']=e[1]/max(e[0],1e-8)
            f[mode+'_xvar']=cov[0,0];f[mode+'_yvar']=cov[1,1]
        within_cov=np.zeros((2,2));dof=0
        for _,gg in g.groupby('t_ms'):
            pp=gg[['x_m','y_m']].to_numpy();pp=pp-pp.mean(0)
            within_cov+=pp.T@pp;dof+=max(len(gg)-1,0)
        within_cov/=max(dof,1);e=np.linalg.eigvalsh(within_cov)
        f['within_eig0'],f['within_eig1']=e
        f['within_xvar']=within_cov[0,0];f['within_yvar']=within_cov[1,1]
        f['within_dof']=dof
        for name,val in [('rcs',r),('velocity',v),('absvelocity',np.abs(v))]:
            f[name+'_skew']=skew(val);f[name+'_kurt']=kurtosis(val)
            f[name+'_iqr']=np.quantile(val,.75)-np.quantile(val,.25)
            f[name+'_mad']=np.median(np.abs(val-np.median(val)))
        rows.append(f)
    extra=pd.DataFrame(rows).set_index('sample_id')
    return pd.concat([basic,extra],axis=1).replace([np.inf,-np.inf],np.nan).fillna(0)


if __name__=='__main__':
    root=Path('competition/q1');train=pd.read_csv(root/'data/train.csv');test=pd.read_csv(root/'data/test.csv')
    geo=sensor_geometry(train)
    X=features(train,geo);Xt=features(test,geo)
    X.to_csv(root/'features_improved.csv');Xt.to_csv(root/'features_test_improved.csv')
    train.groupby('sample_id').label.first().reindex(X.index).to_csv(root/'labels.csv')
    import json
    (root/'sensor_geometry.json').write_text(json.dumps({k:v.tolist() for k,v in geo.items()}))
    print(X.shape,flush=True)
