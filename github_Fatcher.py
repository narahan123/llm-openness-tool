from typing import Dict, Any
import requests

def github_fetcher(repo_full_name: str,
                   branch: str = "main",
                   token: str = None) -> Dict[str, Any]:
    """
    Fetches from a GitHub repo:
      - files: 모든 파일 경로 리스트
      - license_files: 'LICENSE'로 시작하는 모든 파일과 그 내용
      - readme: README.md 내용 (없으면 빈 문자열)
      - py_files: 모든 .py 파일과 그 내용
    """
    headers = {"Accept": "application/vnd.github.v3+json"}
    if token:
        headers["Authorization"] = f"token {token}"

    # 1) 저장소 트리 조회 (recursive)
    tree_url = f"https://api.github.com/repos/{repo_full_name}/git/trees/{branch}?recursive=1"
    resp = requests.get(tree_url, headers=headers)
    resp.raise_for_status()
    tree = resp.json().get("tree", [])

    # 2) 모든 blob 경로 추출
    paths = [item["path"] for item in tree if item["type"] == "blob"]

    # raw 콘텐츠 fetch helper
    def fetch_raw(path: str) -> str:
        raw_url = f"https://raw.githubusercontent.com/{repo_full_name}/{branch}/{path}"
        r = requests.get(raw_url, headers=headers)
        return r.text if r.status_code == 200 else ""

    # 3) LICENSE* 파일 모두 가져오기
    license_paths = [p for p in paths if p.upper().startswith("LICENSE")]
    license_files: Dict[str, str] = {
        p: fetch_raw(p) for p in license_paths
    }

    # 4) README.md 가져오기
    readme = fetch_raw("README.md") if "README.md" in paths else ""

    # 5) .py 파일 모두 가져오기
    py_files: Dict[str, str] = {
        p: fetch_raw(p)
        for p in paths
        if p.endswith(".py")
    }

    return {
        "repo": repo_full_name,
        "branch": branch,
        "files": paths,
        "license_files": license_files,
        "readme": readme,
        "py_files": py_files
    }

# 사용 예시
if __name__ == "__main__":
    info = github_fetcher("google-gemini/gemma-cookbook", branch="main")
    import json
    # print(json.dumps(info, indent=2, ensure_ascii=False))


 # import json
    # print(json.dumps(result, indent=2, ensure_ascii=False))

    for key,values in info.items():
        print("*"*30)
        print(key)
        print(values)