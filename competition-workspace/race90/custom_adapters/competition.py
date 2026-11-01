from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import LinearSVC


def radar_features(data):
    rows = []
    for sid, g in data.groupby('sample_id', sort=True):
        g = g.sort_values('t_ms')
        f = {'sample_id': sid, 'count': len(g), 'times': g.t_ms.nunique(),
             'sensors': g.sensor_id.nunique()}
        cols = ['rcs_dbsm', 'radial_velocity_comp_mps', 'radial_velocity_mps',
                'range_m', 'azimuth_rad', 'x_m', 'y_m']
        for c in cols:
            v = g[c].to_numpy()
            for name, value in [('mean', v.mean()), ('std', v.std()), ('min',v.min()),
                                ('max',v.max()), ('q10',np.quantile(v,.1)),
                                ('median',np.median(v)), ('q90',np.quantile(v,.9))]:
                f[c+'_'+name] = value
        t = g.t_ms.to_numpy()/1000
        tc = t-t.mean()
        denom = np.dot(tc,tc)+1e-12
        residuals=[]
        for c in ['x_m','y_m','rcs_dbsm','radial_velocity_comp_mps']:
            v=g[c].to_numpy(); slope=np.dot(tc,v-v.mean())/denom
            residual=v-v.mean()-slope*tc
            f[c+'_slope']=slope; f[c+'_resid_std']=residual.std()
            f[c+'_resid_qspan']=np.quantile(residual,.9)-np.quantile(residual,.1)
            if c in ['x_m','y_m']: residuals.append(residual)
        eig=np.linalg.eigvalsh(np.cov(residuals))
        f['spatial_eigen_small'],f['spatial_eigen_large']=eig
        f['track_speed']=np.hypot(f['x_m_slope'],f['y_m_slope'])
        f['comp_abs_mean']=g.radial_velocity_comp_mps.abs().mean()
        counts=g.groupby('t_ms').size().to_numpy()
        f['detections_per_time_mean']=counts.mean();f['detections_per_time_std']=counts.std()
        for c in ['x_m','y_m','rcs_dbsm','radial_velocity_comp_mps']:
            s=g.groupby('t_ms')[c].std().fillna(0)
            f[c+'_within_time_std']=s.mean()
        for sensor in [1,2,3,4]:
            sg=g[g.sensor_id==sensor]
            f['sensor_'+str(sensor)+'_fraction']=len(sg)/len(g)
        rows.append(f)
    return pd.DataFrame(rows).set_index('sample_id').fillna(0)


def text_model(c=3.0, ngrams=(2,5)):
    return Pipeline([('features',FeatureUnion([
        ('char',TfidfVectorizer(analyzer='char',ngram_range=ngrams,sublinear_tf=True)),
        ('word',TfidfVectorizer(ngram_range=(1,2),sublinear_tf=True))])),
        ('model',LinearSVC(C=c, max_iter=5000, random_state=42))])


def solve(ctx):
    root=Path(ctx.cfg['data_root'])
    train=pd.read_csv(root/ctx.task['train_csv'])
    test=pd.read_csv(root/ctx.task['test_csv'])
    radar=ctx.task['name']=='q1'
    if radar:
        X=radar_features(train); Xt=radar_features(test)
        y=train.groupby('sample_id').label.first().reindex(X.index).to_numpy()
        ids=Xt.index.to_numpy()
    else:
        X=train.report; Xt=test.report; y=train.fault_type.to_numpy();ids=test.id.to_numpy()
    folds=list(StratifiedKFold(5,shuffle=True,random_state=42).split(np.zeros(len(y)),y))
    options=([('extra_trees',{})] if ctx.stage=='baseline' else
             [('catboost_d4',{'depth':4}),('catboost_d6',{'depth':6}),('catboost_d3',{'depth':3})]) if radar else [('text_svc',{'c':3.0})]
    for name,params in options:
        if ctx.seconds_left()<20: break
        def make_model():
            if name=='extra_trees':
                return ExtraTreesClassifier(n_estimators=400,max_features=.8,n_jobs=ctx.threads,random_state=42)
            if name.startswith('catboost'):
                from catboost import CatBoostClassifier
                return CatBoostClassifier(iterations=1200,learning_rate=.035,loss_function='MultiClass',
                    task_type='GPU' if ctx.device!='cpu' else 'CPU',devices='0',
                    thread_count=ctx.threads,verbose=False,random_seed=42,l2_leaf_reg=5,**params)
            return text_model(**params)
        oof=np.empty(len(y),dtype=object)
        for fold,(a,b) in enumerate(folds):
            m=make_model();m.fit(X.iloc[a],y[a]);oof[b]=np.asarray(m.predict(X.iloc[b])).ravel()
            print(name,'fold',fold,accuracy_score(y[b],oof[b]),flush=True)
        score=accuracy_score(y,oof) if radar else f1_score(y,oof,average='macro')
        m=make_model();m.fit(X,y);pred=np.asarray(m.predict(Xt)).ravel()
        frame=pd.DataFrame({ctx.task['id_column']:ids,ctx.task['target']:pred})
        oofpath=ctx.output/(name+'_oof.csv')
        pd.DataFrame({'truth':y,'prediction':oof}).to_csv(oofpath,index=False)
        artifacts={'oof.csv':oofpath}
        if name.startswith('catboost'):
            modelpath=ctx.output/(name+'.cbm');m.save_model(str(modelpath));artifacts['model.cbm']=modelpath
        ctx.publish(frame,float(score),{'algorithm':name,'validation':'5-fold stratified OOF; sample-level for radar',
                     'refit_all_rows':True,'features':list(X.columns) if radar else 'char + word TF-IDF'},
                    model=None if name.startswith('catboost') else m,artifacts=artifacts)
