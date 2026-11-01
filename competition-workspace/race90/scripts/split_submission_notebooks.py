"""Extract large JSON prediction payloads, keeping the exact existing results."""
import argparse
import ast
import base64
import gzip
import hashlib
import json
import shutil
import zlib
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / "최종제출"
CHECKS = ROOT / "race90/notebook-checks/split"
SOURCE = ROOT / "race90/notebook-checks/final/python397"


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--created-at-utc', required=True)
    args = p.parse_args()
    CHECKS.mkdir(parents=True, exist_ok=True)
    records = []
    for q in (1, 2, 3):
        folder = DEST / (str(q) + "번")
        original = next((SOURCE / ("Problem #" + str(q))).glob('*.ipynb'))
        source = nbformat.read(original, as_version=4)
        sources = []
        data_name = "prediction_data_q%d.json.gz" % q
        data_path = folder / data_name
        for cell in source.cells:
            if cell.cell_type != 'code':
                continue
            value = cell.source
            if '_packed_snapshot = (' in value:
                tree = ast.parse(value)
                node = next(n for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == '_packed_snapshot' for t in n.targets))
                raw = zlib.decompress(base64.b64decode(ast.literal_eval(node.value)))
                payload = json.loads(raw)
                compressed = gzip.compress(raw, compresslevel=9, mtime=0)
                data_path.write_bytes(compressed)
                payload_hash = hashlib.sha256(raw).hexdigest()
                value = (
                    "# 별도 파일에는 예측 산출물과 검증 정보만 들어 있습니다. Python 코드를 실행하지 않습니다.\n"
                    "import gzip\n"
                    "PREDICTION_DATA_PATH = Path.cwd() / " + repr(data_name) + "\n"
                    "if not PREDICTION_DATA_PATH.is_file():\n"
                    "    raise FileNotFoundError('예측 데이터 파일을 Notebook과 같은 폴더에 업로드하세요: ' + str(PREDICTION_DATA_PATH))\n"
                    "with gzip.open(PREDICTION_DATA_PATH, 'rb') as _handle:\n"
                    "    _snapshot_bytes = _handle.read()\n"
                    "if hashlib.sha256(_snapshot_bytes).hexdigest() != " + repr(payload_hash) + ":\n"
                    "    raise ValueError('예측 데이터 파일이 손상되었거나 다른 문항 파일입니다.')\n"
                    "snapshot = json.loads(_snapshot_bytes.decode('utf-8'))\n"
                    "del _snapshot_bytes\n"
                    "print('산출물 스냅샷:', snapshot['created_at_utc'])\n"
                    "print('실행 방식:', snapshot['execution_mode'])"
                )
            ast.parse(value, feature_version=(3, 9))
            sources.append(value)
        title = original.stem
        note = (
            '# ' + title + '\n\n'
            '이 Notebook과 **`' + data_name + '`**을 해당 문제 폴더에 함께 업로드하세요.\n'
            '기존 `data/` 폴더를 사용하며 **Kernel → Restart & Run All**로 `submission.csv`를 생성합니다.\n'
            '데이터 파일은 압축을 풀지 않습니다. 경로가 다르면 첫 코드 셀의 `DATA_ROOT`를 수정하세요.\n\n'
            'Notebook이 열리지 않으면 제공한 **`solution_q' + str(q) + '.py`**의 전체 코드를 원본 Notebook의 코드 셀에 붙여넣고 실행할 수 있습니다.\n\n'
            '**실행 방식:** ' + payload['execution_mode'] + '. 기존 확정 결과를 그대로 사용합니다.\n'
            '1·2번의 신경망을 다시 학습하지 않습니다. 개발 검증 지표는 숨겨진 테스트 점수가 아닙니다.'
        )
        nb = nbformat.v4.new_notebook(cells=[nbformat.v4.new_markdown_cell(note)] + [nbformat.v4.new_code_cell(s) for s in sources])
        nb.nbformat_minor = 2
        for cell in nb.cells:
            cell.pop('id', None)
        nb.metadata = {
            'kernelspec': {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'},
            'language_info': {'name': 'python', 'version': '3.9.7'},
            'race90_split': {'question': q, 'created_at_utc': args.created_at_utc, 'prediction_data_file': data_name, 'predictions_changed': False, 'online_kernel_directly_verified': False},
        }
        nbformat.validate(nb)
        output = folder / original.name
        nbformat.write(nb, output)
        script_path = folder / ('solution_q%d.py' % q)
        script = '# -*- coding: utf-8 -*-\n# 이 전체 코드를 대회 원본 Notebook의 코드 셀에 붙여넣어 실행할 수 있습니다.\n'
        script += '# 함께 제공한 %s 파일을 같은 문제 폴더에 업로드하세요.\n\n' % data_name
        script += '\n\n'.join('# %% 실행 단계 %d\n%s' % (i + 1, text) for i, text in enumerate(sources)) + '\n'
        ast.parse(script, feature_version=(3, 9))
        script_path.write_text(script, encoding='utf-8')
        stage = CHECKS / 'input' / ('Problem #%d' % q)
        stage.mkdir(parents=True, exist_ok=True)
        for path in (output, data_path, script_path):
            shutil.copy2(path, stage / path.name)
        records.append({'question': q, 'notebook': original.name, 'previous_notebook_bytes': original.stat().st_size, 'new_unexecuted_notebook_bytes': output.stat().st_size, 'script_bytes': script_path.stat().st_size, 'data_bytes': data_path.stat().st_size, 'data_sha256': hashlib.sha256(compressed).hexdigest(), 'payload_sha256': payload_hash})
    (CHECKS / 'build-manifest.json').write_text(json.dumps({'created_at_utc': args.created_at_utc, 'notebooks': records}, ensure_ascii=False, indent=2))
    print(json.dumps(records, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
