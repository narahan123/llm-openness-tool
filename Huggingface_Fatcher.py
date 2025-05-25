from typing import Dict, Any
import requests

def huggingface_fetcher(model_id: str) -> Dict[str, Any]:
    base_api = f"https://huggingface.co/api/models/{model_id}?full=true"
    resp = requests.get(base_api)
    resp.raise_for_status()
    data = resp.json()
    #print(data)
    # 1) siblings 목록 수집
    siblings = [f.get("rfilename", "") for f in data.get("siblings", [])]

    
    # helper: raw URL 생성
    def fetch_raw(filename: str) -> str:
        url = f"https://huggingface.co/{model_id}/raw/main/{filename}"
        r = requests.get(url)
        return r.text if r.status_code == 200 else ""
    
    # 2) 주요 파일 콘텐츠 가져오기
    readme = fetch_raw("README.md") if "README.md" in siblings else ""
    config = fetch_raw("config.json") if "config.json" in siblings else ""
    
    # LICENSE 파일 탐색 (LICENSE, LICENSE.md 등)
    license_candidates = [fn for fn in siblings 
                          if fn.upper().startswith("LICENSE")]
    license_file = ""
    if license_candidates:
        # 가장 첫 번째 파일을 사용
        license_file = fetch_raw(license_candidates[0])
    
    # 3) .py 파일 모두 가져오기
    py_files = {}
    for fn in siblings:
        if fn.endswith(".py"):
            py_files[fn] = fetch_raw(fn)
    
    return {
        "model_id": model_id,
        "files": siblings,
        "readme": readme,
        "config": config,
        "license_file": license_file,
        "py_files": py_files
    }

# 사용 
if __name__ == "__main__":
    result = huggingface_fetcher("deepseek-ai/DeepSeek-R1")
    # import json
    # print(json.dumps(result, indent=2, ensure_ascii=False))

    for key,values in result.items():
        print("*"*30)
        print(key)
        print(values)










    """
    Fetches file list and contents of key files for a given Hugging Face model.
    
    Returns a dict with:
    - model_id
    - files: list of sibling filenames
    - readme: content of README.md (or empty if 없음)
    - config: content of config.json (or empty if 없음)
    - license_file: content of LICENSE* files (or empty if 없음)
    - py_files: dict mapping each *.py filename to its content
    """