"""Package small notebooks, plain code, and external JSON prediction data."""
from pathlib import Path
import argparse, ast, gzip, hashlib, json, shutil, zipfile
import nbformat
from package_final_submissions import check_csv, digest, scan_text, preserve_then_copy, ALIASES

ROOT=Path(__file__).resolve().parents[2]
DEST=ROOT/'최종제출'
CHECKS=ROOT/'race90/notebook-checks/split'

def main():
    p=argparse.ArgumentParser();p.add_argument('--completed-at-utc',required=True);args=p.parse_args()
    verified=CHECKS/'verified'
    build=json.loads((CHECKS/'build-manifest.json').read_text())
    execution=json.loads((verified/'execution-report.json').read_text())
    scripts=json.loads((verified/'script-execution-report.json').read_text())
    previous=json.loads((ROOT/'race90/notebook-checks/final/final.json').read_text())
    assert execution['python']=='3.9.7' and execution['numpy']=='1.23.5' and execution['pandas']=='1.5.2'
    assert len(execution['cpu_affinity'])==2 and execution['address_space_limit_bytes']==4*1024**3
    manifest=[]
    for old,run,script,info in zip(previous['notebooks'],execution['notebooks'],scripts,build['notebooks']):
        q=run['question'];assert q==old['question']==script['question']==info['question']
        assert old['csv_sha256']==run['csv_sha256']==script['csv_sha256']
        source=verified/('Problem #'+str(q));folder=DEST/(str(q)+'번')
        name=run['notebook'];nb=nbformat.read(source/name,as_version=4);nbformat.validate(nb)
        assert nb.nbformat_minor==2 and nb.metadata.language_info.version=='3.9.7'
        count=0
        for cell in nb.cells:
            if cell.cell_type!='code':continue
            count+=1;assert cell.execution_count==count
            assert not any(o.output_type=='error' for o in cell.outputs)
            ast.parse(cell.source,feature_version=(3,9))
            assert '_packed_snapshot' not in cell.source and 'base64' not in cell.source
        assert count==run['executed_code_cells']
        scan_text((source/name).read_text(),name)
        raw=gzip.decompress((source/('prediction_data_q%d.json.gz'%q)).read_bytes())
        assert digest(source/('prediction_data_q%d.json.gz'%q))==info['data_sha256']
        assert hashlib.sha256(raw).hexdigest()==info['payload_sha256']
        json.loads(raw);scan_text(raw.decode('utf-8'),name+'::external JSON')
        solution=(source/('solution_q%d.py'%q)).read_text();ast.parse(solution,feature_version=(3,9));scan_text(solution,name+'::solution')
        assert '_packed_snapshot' not in solution and 'base64' not in solution
        # Match each executed code cell to its corresponding plain-text script segment.
        segments=solution.split('# % 실행 단계 ')[1:]
        assert len(segments)==count
        for cell,segment in zip([c for c in nb.cells if c.cell_type=='code'],segments):
            assert segment.split('\n',1)[1].strip()==cell.source.strip()
        check_csv(q,source/'submission.csv')
        for file in source.iterdir():
            if file.is_file() and file.suffix in {'.ipynb','.py','.gz','.csv'}:
                shutil.copy2(file,folder/file.name)
        manifest.append(dict(run,notebook_bytes=(folder/name).stat().st_size,notebook_sha256=digest(folder/name),script='solution_q%d.py'%q,script_sha256=digest(folder/('solution_q%d.py'%q)),data_file='prediction_data_q%d.json.gz'%q,data_bytes=info['data_bytes'],data_sha256=info['data_sha256'],csv_matches_previous=True))

    instructions='''코드·예측 데이터 분리본 실행 안내

기존 확정 예측은 변경하지 않았습니다. 큰 배열을 Notebook 밖의 .json.gz 파일로 옮겼습니다.
각 문항 폴더에 작은 Notebook, solution_qN.py, prediction_data_qN.json.gz, submission.csv가 있습니다.

방법 A: 원본 Notebook에서 코드 파일 실행
1. 해당 문제 폴더에 solution_qN.py와 prediction_data_qN.json.gz를 함께 업로드합니다.
2. 원래 대회 Notebook의 빈 코드 셀에 문항에 맞는 한 줄을 넣고 그 셀을 실행합니다.
   1번: %run ./solution_q1.py
   2번: %run ./solution_q2.py
   3번: %run ./solution_q3.py
3. 같은 폴더에 submission.csv가 생성됩니다. 실행 결과를 확인하고 Notebook을 저장하세요.
   solution_qN.py의 전체 내용을 빈 코드 셀에 직접 붙여넣어 실행해도 됩니다.

방법 B: 작은 Notebook 사용
1. 해당 문제 폴더에 .ipynb와 prediction_data_qN.json.gz를 함께 업로드합니다.
2. Python 3 커널에서 Kernel → Restart & Run All을 실행합니다.
3. 마지막 셀의 저장 완료 메시지를 확인하고 Notebook을 저장하세요.
이 방법에서는 .py 파일이 필수는 아닙니다. 실제 코드는 작은 Notebook에도 모두 들어 있습니다.

prediction_data_qN.json.gz는 압축을 풀지 마세요. 코드가 직접 읽습니다.
기존 data 폴더를 사용합니다. 1·3번은 data/train.csv와 data/test.csv,
2번은 data/aerial_objects가 필요합니다. 데이터 경로가 다르면 DATA_ROOT를 지정하세요.
1번 300행, 2번 300행, 3번 180행이며 세 문항의 CSV 이름은 모두 submission.csv입니다.
문항별 폴더를 구분하여 대회 제출 화면에 올리세요.

실행 방식
1번: 저장한 GPU 확률을 앙상블하여 예측·CSV를 다시 계산합니다.
2번: 저장한 GPU 탐지 결과를 병합·집계하고 기존 이미지 검토 결과를 반영합니다.
3번: 기존 CPU 분류기로 train.csv를 학습하고 test.csv를 실제로 다시 예측합니다.
추가 패키지 설치나 GPU는 필요 없습니다. CPU 재학습 대안으로 모델을 변경하지 않았습니다.
원래 GPU 작업 소스와 큰 Notebook은 로컬 race90/notebook-checks/restore-version에 보존했습니다.

검증
Python 3.9.7, NumPy 1.23.5, pandas 1.5.2, CPU affinity 2개, 주소 공간 상한 4 GiB인
별도 x86_64 환경에서 작은 Notebook 3개와 .py 파일 3개를 모두 실행했습니다.
이 환경에는 PyTorch 1.10.0+cpu도 설치되어 있으나 제출 코드는 해당 API를 사용하지 않습니다.
세 CSV의 SHA256이 이전 확정 제출 결과와 전부 같습니다.
Notebook은 구형 Jupyter 호환성을 위해 nbformat 4.2로 저장했습니다.
실제 대회 온라인 커널 실행과 제출 접수는 직접 확인하지 않았습니다.
'''
    (DEST/'실행안내.txt').write_text(instructions,encoding='utf-8')
    validation={'completed_at_utc':args.completed_at_utc,'format':'small notebook + plain Python code + external gzip-compressed JSON data','submission_files_ready':True,'online_submission_completed':False,'predictions_changed':False,'cpu_retraining_alternative_cancelled_by_user':True,'execution_environment':{k:v for k,v in execution.items() if k!='notebooks'},'notebooks':manifest,'plain_python_execution':scripts,'prompt_injection_scan':'Passed notebook, plain code, and decompressed JSON payload checks; original ZIP audit remains in previous validation report.','no_encoded_predictions_in_notebook_or_python_source':True,'same_three_csv_hashes_as_previous_selected_submission':True}
    (DEST/'검증결과.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2),encoding='utf-8')
    files=sorted(p for p in DEST.rglob('*') if p.is_file() and p.name!='파일목록_SHA256.txt')
    assert len(files)==14,[str(p) for p in files]
    (DEST/'파일목록_SHA256.txt').write_text('\n'.join(digest(p)+'  '+str(p.relative_to(DEST)) for p in files)+'\n')
    archive_path=ROOT/'제출파일_최종.zip'
    with zipfile.ZipFile(archive_path,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for path in sorted(p for p in DEST.rglob('*') if p.is_file()):z.write(path,arcname=str(path.relative_to(DEST)))
    with zipfile.ZipFile(archive_path) as z:
        assert z.testzip() is None and len(z.namelist())==15
        for name in z.namelist():assert z.read(name)==(DEST/name).read_bytes()
    for item in manifest:
        q=item['question'];folder=DEST/(str(q)+'번');target=ROOT/'race90/competition'/('q'+str(q))
        for name in (item['notebook'],item['script'],item['data_file'],'submission.csv'):
            preserve_then_copy(folder/name,target/name,CHECKS/'previous-competition'/('q'+str(q))/name)
        preserve_then_copy(folder/item['notebook'],ROOT/ALIASES[q],CHECKS/'previous-root-notebooks'/ALIASES[q])
        # Root notebook aliases also need their corresponding external data nearby.
        shutil.copy2(folder/item['data_file'],ROOT/item['data_file'])
    final={'completed_at_utc':args.completed_at_utc,'submission_ready':True,'packaged':True,'archive':str(archive_path),'archive_bytes':archive_path.stat().st_size,'archive_sha256':digest(archive_path),'online_submission_completed':False,'predictions_changed':False,'notebooks':manifest}
    (CHECKS/'final.json').write_text(json.dumps(final,ensure_ascii=False,indent=2));print(json.dumps(final,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
