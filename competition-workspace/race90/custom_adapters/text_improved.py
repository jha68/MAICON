import re


def core_report(text):
    # Main observation follows one of the report's structural headers.
    headers = [r'운용자가\s*', r'정비반 보고:\s*', r'기능 시험 결과\s*',
               r'다음 사항을 확인함\.\s*',r'현장 정비 기록: .*?에서\s*',
               r'\[현상\]\s*',r'\[점검 메모\] .*?에서\s*',
               r'정비 메모 — .*?:\s*',r'운용자 인계(?:사항| 기록)\([^)]*\):\s*',
               r'(?:에 대한 )?재현 시험(?:에서| 결과,)\s*',
               r'점검 결과,\s*',r'운용 후 보고:\s*']
    found = []
    for pat in headers:
        m=re.search(pat,text)
        if m:found.append(m)
    if found:
        m=min(found,key=lambda x:x.start());text=text[m.end():]
    # A normal finding may be inserted immediately after the header.
    parts=text.split('.')
    while len(parts)>1 and re.search(r'가능성은 낮으며|특이사항이 없었다|정상으로 확인됐다|뒷받침하는 징후는 발견되지 않았다',parts[0]):
        parts=parts[1:]
    text='.'.join(parts).strip()
    endings = [r'이라고 보고함',r'15분 점검 중',r'정비 후 첫 운용에서',
               r'외기 온도는 정상 범위였음',r'증상은 2회 연속 재현됨',
               r'주변 환경 조건은 정상 범위였다',r'동일 현상이 재시험에서도',
               r'짧은 확인 운전에서도',r'동일 조건에서 두 차례',
               r'정지 후 다시 시작하면',r'\[대상\]',r'\[기타\]']
    text=re.split('|'.join(endings),text)[0]
    text=text.split('.')[0]
    return text.strip(' .')


def semantic_rule(text):
    """Domain-level component rules applied only to the primary observation."""
    s=core_report(text)
    if re.search(r'통신|송신|수신|송수신|신호|무선|안테나|네트워크|호출|채널|헤드셋|음성|통화|단말|메시지|자료 교환|데이터|외부 장비와의 연결',s):
        return 'COMMUNICATION'
    cooling = re.search(r'냉각|라디에이터|온도|과열|팬|풍량|뜨거|열 축적|열 관련|흡입구|공기 흡입부|외기 조건|내부 측정값|보조 탱크|보조 저장조|장시간.*보호 경보|연속 운전.*보호 경보',s)
    electrical_heat = re.search(r'(전원 케이블|전력선).*과열',s)
    if cooling and not electrical_heat:
        return 'COOLING'
    if re.search(r'바퀴|차륜|조향|브레이크|제동|타이어|휠|트레드|직진|방향 전환|차체|속도.*진동.*저속',s):
        return 'MOBILITY'
    if re.search(r'엔진|동력부|공회전|크랭킹|연소|점화|윤활|배기|회전수|가속.*회전|진동감.*가속|첫 기동|기계음|금속성 충격|노킹|회전이 급격히|부하 증가.*힘이 빠|오르막.*출력 부족',s):
        return 'ENGINE'
    if re.search(r'배터리|전압|전류|충전|축전|배선|전력|전기|발전|회로|계기|커넥터|단자|전원|표시등|표시 램프',s):
        return 'ELECTRICAL'
    if re.search(r'압력|실린더|펌프|피팅|작동유|유압|액추에이터|저압|연결부.*젖|저장 탱크|구동.*둔',s):
        return 'HYDRAULIC'
    return None


if __name__=='__main__':
    import numpy as np
    import pandas as pd
    from pathlib import Path
    from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
    from sklearn.metrics import f1_score
    from custom_adapters.competition import text_model
    root=Path('competition/q3')
    train=pd.read_csv(root/'data/train.csv');test=pd.read_csv(root/'data/test.csv')
    train['core']=train.report.map(core_report);test['core']=test.report.map(core_report)
    train.to_csv(root/'train_core.csv',index=False);test.to_csv(root/'test_core.csv',index=False)
    print('Train cores',train.core.nunique(),'test cores',test.core.nunique())
    splits=list(StratifiedGroupKFold(5,shuffle=True,random_state=42).split(train.core,train.fault_type,train.core))
    m=text_model(c=1.0,ngrams=(1,4))
    p=cross_val_predict(m,train.core,train.fault_type,cv=splits)
    print('GroupCV',f1_score(train.fault_type,p,average='macro'))
    m.fit(train.core,train.fault_type)
    scores=m.decision_function(test.core);test['pred']=m.classes_[scores.argmax(1)]
    test['margin']=np.sort(scores,axis=1)[:,-1]-np.sort(scores,axis=1)[:,-2]
    test.to_csv(root/'core_review.csv',index=False)
    for _,r in test.drop_duplicates('core').iterrows():print(r.id,r.pred,round(r.margin,2),r.core)
