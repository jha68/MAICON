"""Outer-fold evaluation with early stopping restricted to inner training data."""
from pathlib import Path
import argparse, gc, json, time
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, log_loss
from tabicl import FinetunedTabICLClassifier

p = argparse.ArgumentParser()
p.add_argument('--device', default='cuda:0')
p.add_argument('--features', default='original')
p.add_argument('--lr', type=float, default=1e-5)
p.add_argument('--epochs', type=int, default=60)
p.add_argument('--seed', type=int, default=42)
p.add_argument('--cv-seed', type=int, default=42)
p.add_argument('--freeze-col', action='store_true')
p.add_argument('--freeze-row', action='store_true')
p.add_argument('--estimators', type=int, default=32)
p.add_argument('--fine-estimators', type=int, default=4)
p.add_argument('--validation-estimators', type=int, default=8)
p.add_argument('--patience', type=int, default=12)
p.add_argument('--tag', default='')
args = p.parse_args()
torch.set_num_threads(4)
root = Path('/workspace/competition-radar')
X = pd.read_csv(root/'features_physics_train.csv', index_col=0)
Xt = pd.read_csv(root/'features_physics_test.csv', index_col=0)
y = pd.read_csv(root/'labels.csv').label.to_numpy()
counts = {'original':79, 'all':165, 'robust':215, 'physics':337}
columns = list(X.columns[:counts[args.features]])
X, Xt = X[columns], Xt[columns]
name = f'finetune_{args.features}_lr{args.lr:g}_s{args.seed}_cv{args.cv_seed}'
if args.freeze_col: name += '_freeze_col'
if args.freeze_row: name += '_freeze_row'
if args.tag: name += '_'+args.tag
out = np.zeros((len(y),5))
test = np.zeros((len(Xt),5))
scores = []
start = time.time()
for fold,(a,b) in enumerate(StratifiedKFold(5,shuffle=True,random_state=args.cv_seed).split(X,y)):
    model_dir = root/'finetuned_models'/name/str(fold)
    model_dir.mkdir(parents=True,exist_ok=True)
    m = FinetunedTabICLClassifier(
        epochs=args.epochs,learning_rate=args.lr,n_estimators_finetune=args.fine_estimators,
        n_estimators_validation=args.validation_estimators,n_estimators_inference=args.estimators,
        validation_split_ratio=.15,early_stopping=True,patience=args.patience,
        eval_metric='log_loss',device=args.device,random_state=args.seed,
        save_interval=0,verbose=True,freeze_col=args.freeze_col,freeze_row=args.freeze_row,
        extra_classifier_kwargs={'n_jobs':4},
    )
    # Outer validation labels are never used for fitting or early stopping.
    m.fit(X.iloc[a],y[a],output_dir=model_dir)
    out[b] = m.predict_proba(X.iloc[b])
    test += m.predict_proba(Xt)/5
    classes = m.classes_
    scores.append(accuracy_score(y[b],classes[out[b].argmax(1)]))
    print(name,'FOLD',fold,scores[-1],'SECONDS',time.time()-start,flush=True)
    np.savez_compressed(root/(name+'_partial.npz'),oof=out,test=test,classes=classes)
    del m
    gc.collect(); torch.cuda.empty_cache()
np.savez_compressed(root/(name+'.npz'),oof=out,test=test,classes=classes)
result={'args':vars(args),'accuracy':accuracy_score(y,classes[out.argmax(1)]),
        'log_loss':log_loss(y,out),'fold_accuracies':scores,'seconds':time.time()-start,
        'validation':'5 outer folds; fine-tuning early stopping uses only internal split of outer training data.'}
(root/(name+'.json')).write_text(json.dumps(result,indent=2))
print(name,'TOTAL',result,flush=True)
