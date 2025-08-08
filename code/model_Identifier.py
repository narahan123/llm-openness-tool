import json
import re
import requests
import os
from pathlib import Path
from urllib.parse import urlparse
from dotenv import load_dotenv
from openai import OpenAI

from huggingface_Fetcher import huggingface_fetcher
from github_Fetcher import github_fetcher
from arxiv_Fetcher import arxiv_fetcher_from_model
# from openness_Evaluator import evaluate_openness
from github_Dispatcher import filter_github_features
from arxiv_Dispatcher import filter_arxiv_features
from huggingface_Dispatcher import filter_hf_features
from openness_Evaluator import evaluate_openness_from_files
from inference import run_inference

import html

# 환경 변수 로드 및 OpenAI 클라이언트 초기화
dotenv_path = os.path.join(os.getcwd(), '.env')
load_dotenv(dotenv_path)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ──────────────────────────────────────────────────────────────
# GPT-4o로 프리트레인(base) 모델 추정
def gpt_detect_base_model(hf_id: str) -> str | None:
    """
    • 입력 모델이 파인튜닝 모델이면 → 프리트레인 모델 ID 반환
    • 이미 프리트레인(=원본) 모델이면 → None 반환
    • 확신 없다면 GPT가 null 반환 → None
    """
    import textwrap

    def _hf_card_readme(mid: str, max_len: int = 12000) -> str:
        try:
            card = requests.get(
                f"https://huggingface.co/api/models/{mid}?full=true"
            ).json().get("cardData", {}) or {}
            txt = (card.get("content") or "")[:max_len]
            for br in ["main", "master"]:
                r = requests.get(f"https://huggingface.co/{mid}/raw/{br}/README.md")
                if r.status_code == 200:
                    txt += "\n\n" + r.text[:max_len]
                    break
            return txt
        except Exception:
            return ""

    prompt_sys = textwrap.dedent(f"""
        당신은 AI 모델 정보를 분석해 ‘프리트레인(원) 모델’을 찾아내는 전문가입니다.

        • 입력 모델 **{hf_id}** 이(가) 파인튜닝 모델일지 모릅니다.
        • 아래에 제공되는 Hugging Face 카드 / README 내용을 읽고
          ➡️ 파생되었을 가능성이 가장 높은 프리트레인 모델 ID를 추정하십시오.
        • 이미 프리트레인 모델이면 null을 반환하십시오.

        ➤ 출력 형식 → JSON 한 줄만! 예:
            {{ "pretrain_model": "bigscience/bloom-560m" }}
          또는
            {{ "pretrain_model": null }}
    """).strip()

    ctx = _hf_card_readme(hf_id)
    if not ctx:
        return None

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": prompt_sys},
                {"role": "user",   "content": ctx}
            ],
            temperature=0
        )
        pred = json.loads(resp.choices[0].message.content)
        pre_id = pred.get("pretrain_model")
        # 검증: 존재하고 입력과 다를 때만 사용
        if pre_id and isinstance(pre_id, str):
            pre_id = pre_id.strip()
            if pre_id.lower() != hf_id.lower() and test_hf_model_exists(pre_id):
                return pre_id
    except Exception as e:
        print("⚠️ GPT 프리트레인 탐지 실패:", e)
    return None
# ──────────────────────────────────────────────────────────────


# 1. 입력 파싱: URL 또는 org/model
def extract_model_info(input_str: str) -> dict:
    platform = None
    organization = model = None
    if input_str.startswith("http"):
        parsed = urlparse(input_str)
        domain = parsed.netloc.lower()
        segments = parsed.path.strip("/").split("/")
        if len(segments) >= 2:
            organization = segments[0]
            model = segments[1].split("?")[0].split("#")[0].replace(".git", "")
            if "huggingface" in domain:
                platform = "huggingface"
            elif "github" in domain:
                platform = "github"
    else:
        parts = input_str.strip().split("/")
        if len(parts) == 2:
            organization, model = parts
            platform = "unknown"
    if not organization or not model:
        raise ValueError("올바른 입력 형식이 아닙니다. 'org/model' 또는 URL을 입력하세요.")
    full_id = f"{organization}/{model}"
    hf_id = full_id.lower()
    return {"platform": platform, "organization": organization,
            "model": model, "full_id": full_id, "hf_id": hf_id}

# 2. 존재 여부 테스트
def test_hf_model_exists(model_id: str) -> bool:
    resp = requests.get(f"https://huggingface.co/api/models/{model_id}")
    return resp.status_code == 200

def test_github_repo_exists(repo: str) -> bool:
    resp = requests.get(f"https://api.github.com/repos/{repo}")
    return resp.status_code == 200

# 3. 링크 파싱 헬퍼 (★ 원문 케이스 유지)
def extract_repo_from_url(url: str) -> str:
    parsed = urlparse(url)
    parts = parsed.path.strip("/").split("/")
    if len(parts) >= 2:
        repo = parts[1].split("?")[0].split("#")[0].replace(".git", "")
        return f"{parts[0]}/{repo}"  # 원문 케이스 유지
    return ""

# 4. HF 페이지 → GitHub (여러 후보 수집 → 소문자 비교/점수화 → 원문으로 반환)
def find_github_in_huggingface(model_id: str) -> str | None:
    """
    HF 모델 카드에서 GitHub 링크 후보들을 전부 모아 점수화 후 가장 그럴듯한 레포를 반환.
    비교/점수화는 소문자 기준, 반환은 원문 케이스 그대로.
    """
    def _extract_repo_from_url_preserve(url: str) -> str | None:
        try:
            p = urlparse(url)
            if "github.com" not in p.netloc.lower():
                return None
            seg = p.path.strip("/").split("/")
            if len(seg) >= 2:
                repo = seg[1].split("?")[0].split("#")[0].replace(".git", "")
                return f"{seg[0]}/{repo}"  # 원문 케이스 유지
        except Exception:
            pass
        return None

    def _tokenize(s: str) -> list[str]:
        s = s.lower().replace("/", " ")
        s = re.sub(r"[^a-z0-9]+", " ", s)
        return [t for t in s.split() if t]

    def _score(repo_lower: str, hf_org: str, toks: list[str]) -> int:
        score = 0
        org, name = repo_lower.split("/", 1)
        if hf_org and org == hf_org.lower():
            score += 5
        for t in toks:
            if t and t in name:
                score += 2
        if any(k in name for k in ["model", "models", "llm"]):
            score += 2
        for k in ["api","client","sdk","demo","website","docs","doc","notebook","colab",
                  "examples","sample","bench","leaderboard","eval","evaluation","convert",
                  "export","deploy","inference","space","slim","angelslim"]:
            if k in name:
                score -= 2
        return score

    try:
        card = requests.get(
            f"https://huggingface.co/api/models/{model_id}?full=true"
        ).json().get("cardData", {}) or {}

        hf_org = model_id.split("/")[0] if "/" in model_id else ""
        toks = _tokenize(model_id)

        # lower → original 매핑
        cand_map: dict[str, str] = {}

        def _add_candidate(rep: str | None):
            if not rep:
                return
            cand_map.setdefault(rep.lower(), rep)  # lower 키로 dedup, 값은 원문

        # 1) links.repository / links.homepage
        for field in ["repository", "homepage"]:
            links = (card.get("links", {}) or {}).get(field)
            if isinstance(links, str):
                _add_candidate(_extract_repo_from_url_preserve(links))
            elif isinstance(links, list):
                for u in links:
                    _add_candidate(_extract_repo_from_url_preserve(str(u)))

        # 2) model card content
        content = card.get("content", "") or ""
        for url in re.findall(r"https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", content):
            _add_candidate(_extract_repo_from_url_preserve(url))

        # 3) raw README
        for br in ["main", "master"]:
            try:
                r = requests.get(f"https://huggingface.co/{model_id}/raw/{br}/README.md")
                if r.status_code == 200:
                    for url in re.findall(r"https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", r.text):
                        _add_candidate(_extract_repo_from_url_preserve(url))
                    break
            except Exception:
                pass

        if not cand_map:
            return None

        # 4) 존재하는 후보만 점수화(비교는 lower), 반환은 원문 케이스
        best_lower, best_score = None, -10**9
        for rep_lower, rep_orig in cand_map.items():
            if not test_github_repo_exists(rep_orig):  # 원문으로 존재 확인
                continue
            s = _score(rep_lower, hf_org, toks)
            if s > best_score:
                best_lower, best_score = rep_lower, s

        return cand_map[best_lower] if best_lower else None
    except Exception:
        return None

# 5. GitHub 페이지 → HF (fallback: raw README and HTML)
def find_huggingface_in_github(repo: str) -> str:
    for fname in ["README.md"]:
        for branch in ["main", "master"]:
            raw_url = f"https://raw.githubusercontent.com/{repo}/{branch}/{fname}"
            try:
                r = requests.get(raw_url)
                if r.status_code == 200:
                    m = re.search(r"https?://huggingface\.co/([\w\-]+/[\w\-\.]+)", r.text, re.IGNORECASE)
                    if m:
                        candidate = m.group(1).lower()
                        if not candidate.startswith('collections/'):
                            return candidate
                    m2 = re.search(r"huggingface\.co/([\w\-]+/[\w\-\.]+)", r.text, re.IGNORECASE)
                    if m2:
                        candidate = m2.group(1).lower()
                        if not candidate.startswith('collections/'):
                            return candidate
                    m_md = re.search(r"\\[.*?\\]\\((https?://huggingface\.co/[\w\-/\.]+)\\)", r.text, re.IGNORECASE)
                    if m_md:
                        candidate = extract_model_info(m_md.group(1))["hf_id"]
                        if not candidate.startswith('collections/'):
                            return candidate
                    m_html = re.search(r'<a\\s+href="https?://huggingface\.co/([\w\-/\.]+)"', r.text, re.IGNORECASE)
                    if m_html:
                        candidate = m_html.group(1).lower()
                        if not candidate.startswith('collections/'):
                            return candidate
            except:
                pass
    try:
        html = requests.get(f"https://github.com/{repo}").text
        m3 = re.findall(r"https://huggingface\.co/[\w\-]+/[\w\-\.]+", html, re.IGNORECASE)
        for link in m3:
            if 'href' in html[html.find(link)-20:html.find(link)]:
                candidate = extract_model_info(link)["hf_id"]
                if not candidate.startswith('collections/'):
                    return candidate
    except:
        pass
    return None

def gpt_guess_github_from_huggingface(hf_id: str) -> str:
    prompt = f"""
Hugging Face에 등록된 모델 '{hf_id}'에 대해, 이 모델의 원본 코드가 저장된 GitHub 저장소를 추정하세요.

🟢 지켜야 할 규칙:
1. 'organization/repo' 형식으로 **정확한 GitHub 경로만** 반환하세요 (링크 X, 설명 X).
2. 'google-research/google-research'처럼 너무 일반적인 모노리포지터리는 피하고, 모델 단위 저장소가 있다면 그것을 우선 추정하세요.
3. distill 모델인 경우 부모 모델을 찾아주세요.
4. 해당 모델의 이름, 구조, 논문, tokenizer, 사용 라이브러리(PyTorch, JAX, T5 등)를 참고해서 정확한 repo를 추정하세요.
5. 결과는 **딱 한 줄**, 예: `facebookresearch/llama`

🔴 출력에는 부가 설명 없이 GitHub 저장소 경로만 포함해야 합니다.
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        guess = response.choices[0].message.content.strip()
        if "/" in guess:
            return guess
    except Exception as e:
        print("⚠️ GPT HF→GH 추정 중 오류 발생:", e)
    return None

def gpt_guess_huggingface_from_github(gh_id: str) -> str:
    prompt = f"""
Hugging Face에 등록된 모델 '{gh_id}'의 원본 코드가 저장된 Hugging Face 모델 ID를 추정해주세요.
- 정확한 organization/repository 경로만 출력해주세요.
- 모델 이름이나 관련 논문에서 유래된 GitHub 저장소를 기준으로 추정하세요.
- 예시 출력: facebookresearch/llama
- 'google-research/google-research'처럼 광범위한 모노리포지터리는 피하고, 해당 모델을 위한 별도 저장소가 있다면 그쪽을 우선 고려하세요.
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        guess = response.choices[0].message.content.strip().lower()
        if "/" in guess:
            return guess
    except Exception as e:
        print("⚠️ GPT GH→HF 추정 중 오류 발생:", e)
    return None

def run_all_fetchers(user_input: str):
    outdir = make_model_dir(user_input)
    print(f"📁 출력 경로: {outdir}")
    info = extract_model_info(user_input)
    hf_id = gh_id = None
    found_rank_hf = found_rank_gh = None
    full = info['full_id']
    hf_cand = info['hf_id']

    hf_ok = test_hf_model_exists(hf_cand)
    gh_ok = test_github_repo_exists(full)
    print(f"1️⃣ HF: {hf_ok}, GH: {gh_ok}")

    if hf_ok:
        hf_id = hf_cand
        found_rank_hf = 1
    if gh_ok:
        gh_id = full
        found_rank_gh = 1

    if hf_ok and not gh_id:
        gh_link = find_github_in_huggingface(hf_cand)
        print(f"🔍 2순위 HF→GH link: {gh_link}")
        if gh_link and test_github_repo_exists(gh_link):  # 원문 케이스 그대로 사용
            gh_id = gh_link
            found_rank_gh = 2

    if gh_ok and not hf_id:
        hf_link = find_huggingface_in_github(full)
        print(f"🔍 2순위 GH→HF link: {hf_link}")
        if hf_link and test_hf_model_exists(hf_link):
            hf_id = hf_link
            found_rank_hf = 2

    if hf_ok and not gh_id:
        guess_gh = gpt_guess_github_from_huggingface(hf_cand)
        print(f"⏳ 3순위 GPT HF→GH guess: {guess_gh}")
        if guess_gh and test_github_repo_exists(guess_gh):
            gh_id = guess_gh
            found_rank_gh = 3
            print("⚠️ GPT 추정 결과입니다. 저장소가 실제로 해당 모델을 포함하는지 검토 필요")

    if gh_ok and not hf_id:
        guess_hf = gpt_guess_huggingface_from_github(full)
        print(f"⏳ 3순위 GPT GH→HF guess: {guess_hf}")
        if guess_hf and test_hf_model_exists(guess_hf):
            hf_id = guess_hf
            found_rank_hf = 3
            print("⚠️ GPT 추정 결과입니다. 모델 ID가 정확한지 검토 필요")

    if hf_id:
        rank_hf = found_rank_hf or '없음'
        print(f"✅ HF model: {hf_id} (발견: {rank_hf}순위)")
        data = huggingface_fetcher(hf_id, save_to_file=True, output_dir=outdir)
        arxiv_fetcher_from_model(hf_id, save_to_file=True, output_dir=outdir)
        try:
            hf_filtered = filter_hf_features(hf_id, output_dir=outdir)
        except FileNotFoundError:
            hf_filtered = {}
            print("⚠️ HuggingFace JSON 파일이 존재하지 않아 필터링 생략")
        try:
            ax_filtered = filter_arxiv_features(hf_id, output_dir=outdir)
        except FileNotFoundError:
            ax_filtered = {}
            print("⚠️ arXiv JSON 파일이 존재하지 않아 필터링 생략")
    else:
        print("⚠️ HuggingFace 정보 없음")
    if gh_id:
        rank_gh = found_rank_gh or '없음'
        print(f"✅ GH repo: {gh_id} (발견: {rank_gh}순위)")

        try:
            github_fetcher(gh_id, branch="main", save_to_file=True, output_dir=outdir)
        except requests.exceptions.HTTPError:
            print("⚠️ main 브랜치 접근 실패, master 브랜치로 재시도...")
            try:
                github_fetcher(gh_id, branch="master", save_to_file=True, output_dir=outdir)
            except Exception as e:
                print("❌ master 브랜치도 실패:", e)

        try:
            gh_filtered = filter_github_features(gh_id, output_dir=outdir)
        except FileNotFoundError:
            gh_filtered = {}
            print("⚠️ GitHub JSON 파일이 존재하지 않아 필터링 생략")
    else:
        print("⚠️ GitHub 정보 없음")

    # ─── GPT 기반 프리트레인 모델 탐지 + 파이프라인 ───────────────
    base_model_id = gpt_detect_base_model(hf_id) if hf_id else None
    if base_model_id:
        print(f"🧱 GPT가 찾은 프리트레인 모델: {base_model_id}")

        # 1) Hugging Face fetch/dispatch
        huggingface_fetcher(base_model_id, save_to_file=True, output_dir=outdir)
        from pretrain_hf_Dispatcher import filter_pretrain_hf
        filter_pretrain_hf(base_model_id, output_dir=outdir)

        # 2) GitHub (있을 때만)
        base_gh = find_github_in_huggingface(base_model_id)
        if base_gh:
            try:
                github_fetcher(base_gh, save_to_file=True, output_dir=outdir)
                from pretrain_github_Dispatcher import filter_pretrain_gh
                filter_pretrain_gh(base_gh, output_dir=outdir)
            except Exception as e:
                print("⚠️ GH fetch/dispatch 실패:", e)
        else:
            print("⚠️ 프리트레인 모델의 GitHub 레포를 찾지 못해 GH fetcher 건너뜀")

        # 3) arXiv (있을 때만)
        try:
            ax_ok = arxiv_fetcher_from_model(base_model_id,
                                             save_to_file=True,
                                             output_dir=outdir)
            if ax_ok:
                from pretrain_arxiv_Dispatcher import filter_pretrain_arxiv
                filter_pretrain_arxiv(base_model_id, output_dir=outdir)
            else:
                print("⚠️ 논문 링크를 찾지 못해 arXiv fetcher 건너뜀")
        except Exception as e:
            print("⚠️ arXiv fetch/dispatch 실패:", e)
    else:
        base_model_id = None  # GPT가 null 반환 → 프리트레인 없음


    # 8. Openness 평가 수행
    try:
        print("📝 개방성 평가 시작...")
        eval_res = evaluate_openness_from_files(
            full,
            base_dir=str(outdir),
            base_model_id=base_model_id        # ← 인자 추가
        )
        base = full.replace("/", "_")
        outfile = Path(outdir) / f"openness_score_{base}.json"
        print(f"✅ 개방성 평가 완료.  결과 파일: {outfile}")
    except Exception as e:
        print("⚠️ 개방성 평가 중 오류 발생:", e)

    # README 기반 추론은 data가 있을 때만
    if 'data' in locals() and isinstance(data, dict) and data.get("readme"):
        run_inference(data.get("readme"))

def make_model_dir(user_input: str) -> Path:
    info = extract_model_info(user_input)
    base = info["hf_id"]                         # ex) 'bigscience/bloomz-560m'
    safe = re.sub(r'[<>:"/\\|?*\s]+', "_", base) # ex) 'bigscience_bloomz-560m'
    path = Path(safe)
    path.mkdir(parents=True, exist_ok=True)
    return path
###################################################################
# if __name__ == "__main__":
#     user_input = input("🌐 HF/GH URL 또는 org/model: ").strip()
#     model_dir = make_model_dir(user_input)
#     print(f"📁 생성/사용할 폴더: {model_dir}")
#     run_all_fetchers(user_input)

#     info = extract_model_info(user_input)
#     hf_id = info['hf_id']

#     if test_hf_model_exists(hf_id):
#         with open(model_dir / "identified_model.txt", "w", encoding="utf-8") as f:
#             f.write(hf_id)
#         print(f"✅ 모델 ID 저장 완료: {model_dir / 'identified_model.txt'}")
#######################################################################
if __name__ == "__main__":
    try:
        n = int(input("🔢 돌릴 모델 개수: ").strip())
    except ValueError:
        print("숫자를 입력하세요."); exit(1)

    models: list[str] = []
    for i in range(1, n + 1):
        m = input(f"[{i}/{n}] 🌐 HF/GH URL 또는 org/model: ").strip()
        if m:
            models.append(m)

    print("\n🚀 총", len(models), "개 모델을 순차 처리합니다.\n")

    for idx, user_input in enumerate(models, 1):
        print(f"\n======== {idx}/{len(models)} ▶ {user_input} ========")
        try:
            model_dir = make_model_dir(user_input)
            print(f"📁 생성/사용할 폴더: {model_dir}")
            run_all_fetchers(user_input)

            info  = extract_model_info(user_input)
            hf_id = info["hf_id"]
            if test_hf_model_exists(hf_id):
                with open(model_dir / "identified_model.txt", "w", encoding="utf-8") as f:
                    f.write(hf_id)
                print(f"✅ 모델 ID 저장 완료: {model_dir / 'identified_model.txt'}")

        except Exception as e:
            print("❌ 처리 중 오류 발생:", e)
            # 에러 로그 남기고 다음 모델로 계속
            continue

    print("\n🎉 모든 작업이 끝났습니다.")


    # ✅ 바로 추론까지 실행
    prompt = input("📝 프롬프트를 입력하세요: ")
    run_inference(hf_id, prompt)
