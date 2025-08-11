#!/usr/bin/env python
# pretrain_finder.py
# ---------------------------------------------------------
# ▶︎ 사용법
#     python pretrain_finder.py <org/model>
# ▶︎ 결과
#     ./pretrain_candidates_<org>_<model>.json 생성
#     {
#       "input_model": "bigscience/bloomz-560m",
#       "candidates": [
#         { "model_id": "bigscience/bloom-560m", "score": 23 },
#         { "model_id": "google/mt5-small",      "score": 14 }
#       ]
#     }
# ---------------------------------------------------------

import sys, re, json, requests, html, os
from pathlib import Path

# ───────────────────────────────────────── helpers ─────────────────────────────
def _exists(hf_id: str) -> bool:
    try:
        return requests.get(f"https://huggingface.co/api/models/{hf_id}").status_code == 200
    except Exception:
        return False

def _links(text: str) -> list[tuple[str, str]]:
    if not text:
        return []
    out = []
    pat = r"https?://huggingface\.co/([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)"
    for m in re.finditer(pat, text):
        mid = m.group(1)
        ctx = text[max(0, m.start() - 150): m.end() + 60]
        out.append((mid, ctx))
    return out

def _score(mid: str, ctx: str) -> int:
    ctx_l = ctx.lower()
    score = 0
    for kw in ["base model", "pretrained", "original pretrained",
               "checkpoint", "finetuned from", "fine-tuned from"]:
        if kw in ctx_l: score += 5
    for neg in ["instruction", "rlhf", "sft", "chat", "mt"]:
        if neg in ctx_l: score -= 1
    return score

# ─────────────────────────────────────── main logic ───────────────────────────
def find_pretrain_candidates(hf_id: str) -> dict[str, int]:
    """모델 카드·README를 스캔해 후보 프리트레인 모델과 점수 반환"""
    try:
        r = requests.get(f"https://huggingface.co/api/models/{hf_id}?full=true")
        r.raise_for_status()
        card = r.json().get("cardData", {}) or {}
    except Exception as e:
        print("⚠️ HF API 호출 실패:", e)
        return {}

    cands: dict[str, int] = {}

    # 1) cardData.base_model
    bm = card.get("base_model")
    if isinstance(bm, str) and "/" in bm:
        cands[bm] = 10
    elif isinstance(bm, list):
        for x in bm:
            if isinstance(x, str) and "/" in x:
                cands[x] = 10

    # 2) card content
    for mid, ctx in _links(card.get("content", "") or ""):
        cands[mid] = max(cands.get(mid, 0), _score(mid, ctx))

    # 3) raw README
    for br in ["main", "master"]:
        try:
            rr = requests.get(f"https://huggingface.co/{hf_id}/raw/{br}/README.md")
            if rr.status_code == 200:
                for mid, ctx in _links(rr.text):
                    cands[mid] = max(cands.get(mid, 0), _score(mid, ctx))
                break
        except Exception:
            pass

    # 존재 확인 & 정렬
    final = {mid: sc for mid, sc in cands.items()
             if mid.lower() != hf_id.lower() and _exists(mid)}
    return dict(sorted(final.items(), key=lambda x: x[1], reverse=True))

# ────────────────────────────────────────── CLI ───────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("사용법: python pretrain_finder.py <org/model>")
        sys.exit(1)

    input_model = sys.argv[1].strip()
    cand_scores = find_pretrain_candidates(input_model)

    base = input_model.replace("/", "_")
    out_path = Path(f"pretrain_candidates_{base}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "input_model": input_model,
            "candidates": [
                {"model_id": mid, "score": sc} for mid, sc in cand_scores.items()
            ]
        }, f, ensure_ascii=False, indent=2)

    print(f"✅ 후보 JSON 저장 완료: {out_path}")
