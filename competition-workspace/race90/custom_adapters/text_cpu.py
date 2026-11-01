"""Reproducible CPU classifier: primary-symptom rules + trained TF-IDF retrieval."""
import math
import re
from collections import Counter,defaultdict
from custom_adapters.text_improved import core_report,semantic_rule

def grams(text):
    text=re.sub(r'\s+',' ',text)
    return Counter(text[i:i+n] for n in (2,3,4) for i in range(len(text)-n+1))

class ReportClassifier:
    def fit(self,reports,labels):
        unique={}
        for text,label in zip(reports,labels):
            core=core_report(text)
            if core in unique and unique[core]!=label:raise ValueError('Conflicting symptom labels')
            unique[core]=label
        self.cores=list(unique);self.labels=[unique[c] for c in self.cores]
        docs=[grams(c) for c in self.cores];df=Counter()
        for d in docs:df.update(d.keys())
        self.idf={g:math.log((1+len(docs))/(1+n))+1 for g,n in df.items()}
        self.vectors=[self.vector(d) for d in docs]
        return self
    def vector(self,counts):
        x={g:(1+math.log(n))*self.idf[g] for g,n in counts.items() if g in self.idf}
        norm=math.sqrt(sum(v*v for v in x.values())) or 1
        return {g:v/norm for g,v in x.items()}
    def predict_one(self,report):
        rule=semantic_rule(report)
        if rule:return rule
        x=self.vector(grams(core_report(report)))
        sims=[sum(v*vec.get(g,0) for g,v in x.items()) for vec in self.vectors]
        return self.labels[max(range(len(sims)),key=sims.__getitem__)]
    def predict(self,reports):return [self.predict_one(s) for s in reports]

if __name__=='__main__':
    import csv,json
    from pathlib import Path
    root=Path('competition/q3')
    with (root/'data/train.csv').open(encoding='utf-8-sig',newline='') as f:train=list(csv.DictReader(f))
    with (root/'data/test.csv').open(encoding='utf-8-sig',newline='') as f:test=list(csv.DictReader(f))
    model=ReportClassifier().fit([r['report'] for r in train],[r['fault_type'] for r in train])
    predictions=model.predict([r['report'] for r in test])
    with (root/'submission.cpu.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.writer(f);w.writerow(['id','fault_type']);w.writerows((r['id'],p) for r,p in zip(test,predictions))
    print('Saved',len(predictions),'predictions')
    expected=list(csv.DictReader((root/'submission.hybrid.csv').open()))
    print('Differences from E5 + semantic rules:',[(r['id'],p,e['fault_type']) for r,p,e in zip(test,predictions,expected) if p!=e['fault_type']])
