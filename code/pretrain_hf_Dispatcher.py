# pretrain_hf_Dispatcher.py
"""
3-1(사전학습 방법) / 4-1(사전학습 데이터) 전용 HF 디스패처
입력: huggingface_{base}.json (readme 포함)
출력: pretrain_hf_{base}.json  { "pretrain_method": str, "pretrain_data": str, "__evidence": [...] }
"""
import os, json
from typing import Dict, Any, List
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

CHUNK_CHARS   = 60_000
CHUNK_OVERLAP = 2_000
MODEL_NAME    = os.getenv("OPENAI_MODEL_HF_DISPATCHER", "o3-mini")

load_dotenv()
_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM = (
    "당신은 AI 모델 평가 보조 도우미입니다. 사용자가 제공한 허깅페이스 원문에서 "
    "모델의 사전학습 방법(어떻게 pre-training 했는지)과 사전학습 데이터(무엇으로 했는지)를 "
    "한국어로 요약해 **반드시 JSON 객체**로만 출력하세요. "
    '형식: {"pretrain_method": str, "pretrain_data": str, "__evidence": [str, ...]}'
)

def _chunk(t: str) -> List[str]:
    out=[]; n=len(t); i=0
    while i<n:
        end=min(i+CHUNK_CHARS,n)
        out.append(t[i:end])
        if end==n: break
        i=end-CHUNK_OVERLAP
    return out

def _extract_context(j: Dict[str, Any]) -> str:
    # 우리 fetcher 포맷: readme / files ...
    # 혹시 cardData.content가 있으면 같이 사용
    parts=[]
    readme = j.get("readme") or ""
    if isinstance(readme, str) and readme.strip():
        parts.append(readme)
    card = j.get("cardData") or {}
    if isinstance(card, dict):
        c = card.get("content") or ""
        if isinstance(c, str) and c.strip():
            parts.append(c)
    return ("\n\n".join(parts))[:240_000]

def filter_pretrain_hf(model_id: str, save: bool = True, output_dir: str | Path = ".") -> Dict[str, Any]:
    base = model_id.replace("/", "_").lower()
    path_in = Path(output_dir) / f"huggingface_{base}.json"
    if not path_in.exists():
        alt = Path(f"huggingface_{base}.json")
        if not alt.exists():
            raise FileNotFoundError(str(path_in))
        path_in = alt

    hf_json = json.load(open(path_in, encoding="utf-8"))
    ctx = _extract_context(hf_json)

    if not ctx.strip():
        result = {"pretrain_method": "정보 없음", "pretrain_data": "정보 없음", "__evidence": []}
    else:
        results=[]
        for i, ch in enumerate(_chunk(ctx), 1):
            try:
                rsp = _client.chat.completions.create(
                    model=MODEL_NAME,
                    reasoning_effort="medium",
                    response_format={"type":"json_object"},
                    messages=[
                        {"role":"system","content":SYSTEM},
                        {"role":"user","content":f"아래 원문을 읽고 JSON으로만 답하세요.\n\n{ch}"}
                    ],
                )
                results.append(json.loads(rsp.choices[0].message.content))
            except Exception as e:
                print(f"⚠️ HF-pretrain chunk {i} 실패:", e)

        # 병합 규칙: 첫 JSON을 베이스로 채우고, evidence는 합침
        if results:
            base_r = {"pretrain_method":"정보 없음","pretrain_data":"정보 없음","__evidence":[]}
            for r in results:
                if r.get("pretrain_method"): base_r["pretrain_method"]=r["pretrain_method"]
                if r.get("pretrain_data"):   base_r["pretrain_data"]=r["pretrain_data"]
                if isinstance(r.get("__evidence"), list):
                    base_r["__evidence"].extend(r["__evidence"])
            result = base_r
        else:
            result = {"pretrain_method":"정보 없음","pretrain_data":"정보 없음","__evidence":[]}

    out = Path(output_dir) / f"pretrain_hf_{base}.json"
    if save:
        json.dump(result, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print(f"✅ {out} 저장")
    return result
