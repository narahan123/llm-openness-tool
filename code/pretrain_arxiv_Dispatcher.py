# pretrain_arxiv_Dispatcher.py
"""
arXiv 논문 전체 텍스트에서 사전학습 방법/데이터 추출
입력: arxiv_fulltext_{base}.json  또는 arxiv_{base}.json
출력: pretrain_arxiv_{base}.json
"""
import os, json, re
from pathlib import Path
from typing import Dict, List, Optional
from dotenv import load_dotenv
from openai import OpenAI

CHUNK_CHARS   = 60_000
CHUNK_OVERLAP = 2_000
MODEL_NAME    = os.getenv("OPENAI_MODEL_ARXIV_DISPATCHER", "o3-mini")

load_dotenv(); _cli = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYS = (
  "다음 논문 전문에서 모델의 사전학습 방법과 데이터 정보를 찾아 "
  '한국어로 **반드시 JSON 객체**로만 반환하세요. '
  '형식: {"pretrain_method": str, "pretrain_data": str, "__evidence":[str,...]}'
)

def _chunk(text: str) -> List[str]:
    out=[]; n=len(text); i=0
    while i<n:
        end=min(i+CHUNK_CHARS,n); out.append(text[i:end])
        if end==n: break
        i=end-CHUNK_OVERLAP
    return out

def _find_full(base: str, root: Path) -> Optional[Path]:
    for name in (f"arxiv_fulltext_{base}.json", f"arxiv_{base}.json"):
        p = root / name
        if p.exists(): return p
    return None

def _load_fulltext(p: Path) -> str:
    j = json.load(open(p, encoding="utf-8"))
    # 우리 저장 포맷: full_texts(list) 또는 full_text(str)
    if isinstance(j.get("full_texts"), list):
        texts = [t.get("full_text","") if isinstance(t, dict) else str(t) for t in j["full_texts"]]
        return "\n\n".join(texts)
    return j.get("full_text","")

def filter_pretrain_arxiv(model_id: str, save: bool=True, output_dir: str | Path = ".") -> Dict:
    base=model_id.replace("/","_").lower()
    root=Path(output_dir)
    p_in=_find_full(base, root) or _find_full(base, Path("."))
    if not p_in:
        print("⚠️ arXiv json 없음"); return {}
    txt=_load_fulltext(p_in)

    # pre-training 섹션 우선
    m = re.search(r"(?is)(pre[- ]?training|pretraining).*", txt)
    target = m.group(0) if m else txt

    results=[]
    for i, ch in enumerate(_chunk(target), 1):
        try:
            rsp=_cli.chat.completions.create(
                model=MODEL_NAME,
                reasoning_effort="medium",
                response_format={"type":"json_object"},
                messages=[
                    {"role":"system","content":SYS},
                    {"role":"user","content":f"아래 원문을 읽고 JSON으로만 답하세요.\n\n{ch}"}
                ]
            )
            results.append(json.loads(rsp.choices[0].message.content))
        except Exception as e:
            print(f"⚠️ arXiv-pretrain chunk {i} 실패:", e)

    out = {"pretrain_method":"정보 없음","pretrain_data":"정보 없음","__evidence":[]}
    for r in results:
        if r.get("pretrain_method"): out["pretrain_method"]=r["pretrain_method"]
        if r.get("pretrain_data"):   out["pretrain_data"]=r["pretrain_data"]
        if isinstance(r.get("__evidence"), list): out["__evidence"].extend(r["__evidence"])

    p_out = root / f"pretrain_arxiv_{base}.json"
    if save:
        json.dump(out, open(p_out,"w",encoding="utf-8"), ensure_ascii=False, indent=2)
        print(f"✅ {p_out} 저장")
    return out
