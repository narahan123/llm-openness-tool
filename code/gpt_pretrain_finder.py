#!/usr/bin/env python
# gpt_pretrain_finder.py
# ----------------------------------------------------------
# ▶ 사용법
#     python gpt_pretrain_finder.py <org/model>
# ----------------------------------------------------------

import sys, os, json, requests, textwrap
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

# ── 환경 변수에서 OPENAI_API_KEY 불러오기 ───────────────────
load_dotenv()
cli = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ── HF 카드 + README 텍스트 가져오기 ─────────────────────────
def _get_hf_context(hf_id: str, max_len: int = 12000) -> str:
    try:
        card = requests.get(
            f"https://huggingface.co/api/models/{hf_id}?full=true"
        ).json().get("cardData", {}) or {}
        txt = (card.get("content") or "")[:max_len]

        # README raw 추가
        for br in ["main", "master"]:
            r = requests.get(f"https://huggingface.co/{hf_id}/raw/{br}/README.md")
            if r.status_code == 200:
                txt += "\n\n" + r.text[:max_len]
                break
        return txt
    except Exception as e:
        print("⚠️ HF 카드 로딩 실패:", e)
        return ""

# ── GPT-4 프롬프트 ──────────────────────────────────────────
PROMPT_TMPL = textwrap.dedent("""
    당신은 AI 모델 정보를 분석해 ‘프리트레인(원) 모델’을 찾아내는 전문가입니다.

    • 입력으로 제공된 Hugging Face 모델 **{hf_id}** 는 파인튜닝(혹은 체크포인트) 모델일 수도 있습니다.
    • 아래에 제공되는 Hugging Face 카드 / README 내용을 분석해,
      “이 모델이 어떤 프리트레인 모델(checkpoint)에서 파생되었는지” 추정하세요.
    • 답변 형식은 **JSON 한 줄**로만, 예시는 다음과 같습니다.
        {{ "pretrain_model": "bigscience/bloom-560m" }}
      (모델 ID를 소문자 그대로, 링크·주석·백틱 없이!)

    만약 확신할 수 없으면:
        {{ "pretrain_model": null }}
""").strip()


def gpt_find_pretrain(hf_id: str) -> str | None:
    ctx = _get_hf_context(hf_id)
    if not ctx:
        return None

    messages = [
        {"role": "system", "content": PROMPT_TMPL.format(hf_id=hf_id)},
        {"role": "user", "content": ctx}
    ]
    rsp = cli.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},
        messages=messages,
        temperature=0
    )
    try:
        pred = json.loads(rsp.choices[0].message.content)
        return pred.get("pretrain_model")
    except Exception as e:
        print("⚠️ GPT 응답 파싱 실패:", e)
        return None

# ── CLI 진입점 ───────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("사용법: python gpt_pretrain_finder.py <org/model>")
        sys.exit(1)

    model_id = sys.argv[1].strip()
    pre_id = gpt_find_pretrain(model_id)

    base = model_id.replace("/", "_")
    out = Path(f"pretrain_gpt_{base}.json")
    json.dump({"input_model": model_id, "pretrain_model": pre_id},
              open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    if pre_id:
        print(f"✅ GPT-4가 추정한 프리트레인 모델: {pre_id}")
    else:
        print("⚠️ GPT-4가 프리트레인 모델을 확신하지 못했습니다.")
    print("📝 결과 저장:", out)
