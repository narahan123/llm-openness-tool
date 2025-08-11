# pretrain_github_Dispatcher.py
"""
GitHub README / docs에서 사전학습 정보 추출
입력 : github_{base}.json
출력 : pretrain_gh_{base}.json
"""
import os, json, time
from typing import Dict, Any, List
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

CHUNK_CHARS   = 60_000
CHUNK_OVERLAP = 2_000
MODEL_NAME    = os.getenv("OPENAI_MODEL_HF_DISPATCHER", "o3-mini")

load_dotenv(); _cli = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYS = (
  "주어진 GitHub README/문서에서 모델의 사전학습 방법과 "
  "사전학습 데이터 정보를 한국어로 요약해 JSON 형식으로 반환하세요."
)

def _chunk(text: str) -> List[str]:
    n, i, out = len(text), 0, []
    while i < n:
        end = min(i + CHUNK_CHARS, n)
        out.append(text[i:end])
        if end == n: break
        i = end - CHUNK_OVERLAP
    return out

def filter_pretrain_gh(model_id: str,
                       save: bool = True,
                       output_dir: str | Path = ".") -> Dict:
    base = model_id.replace("/", "_").lower()
    root = Path(output_dir)
    pin  = root / f"github_{base}.json"
    if not pin.exists():
        print("⚠️ GH JSON 없음:", pin); return {}

    readme = json.load(open(pin, encoding="utf-8")).get("readme", "")
    results = []
    for idx, ch in enumerate(_chunk(readme), 1):
        print(f"⏳ GH-pretrain chunk {idx} 호출…")
        try:
            rsp = _cli.chat.completions.create(
                model=MODEL_NAME,
                reasoning_effort="medium",
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": SYS},
                    {"role": "user",   "content": ch}
                ]
            )
            results.append(json.loads(rsp.choices[0].message.content))
        except Exception as e:
            print("⚠️ GPT 실패(무시):", e)

    final = max(results, key=lambda d: len(str(d)), default={})
    out = root / f"pretrain_gh_{base}.json"
    if save:
        json.dump(final, open(out, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
        print("✅", out, "저장")
    return final
