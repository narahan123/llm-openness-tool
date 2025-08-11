# inference.py
import os, sys, json, re, shlex, subprocess
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv
from openai import OpenAI

# ─────────────────────────────────────────
# 환경
# ─────────────────────────────────────────
load_dotenv()
_api_key = os.getenv("OPENAI_API_KEY")
if not _api_key:
    raise RuntimeError("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
_client = OpenAI(api_key=_api_key)

OPENAI_MODEL = os.getenv("OPENAI_MODEL_INFER_PICK", "gpt-4o-mini")
ENC_RUN = "cp949" if os.name == "nt" else "utf-8"  # 콘솔 출력 인코딩

# 모델 파이프라인에서 내려주는 기본 출력 폴더 (없으면 현재 폴더)
_DEFAULT_OUTDIR = os.getenv("MODEL_OUTPUT_DIR") or os.getenv("CURRENT_MODEL_DIR") or "."

# ─────────────────────────────────────────
# GPT 프롬프트: 실행 예제 + pip 설치 명령을 JSON으로
# ─────────────────────────────────────────
_SYSTEM = """
너는 Hugging Face README 분석 도우미다.
목표: "로컬에서 바로 돌릴 수 있는 단일 파이썬 예제" 1개와, 실행 전에 필요한 pip 설치 명령을 뽑아라.

규칙:
- 오직 JSON 객체만 출력한다.
- 서버/REST API/vLLM 서버 띄우기 등은 제외하고, "로컬 파이썬 스크립트" 예제를 우선한다.
- 가능하면 transformers 기반의 간단한 chat/inference 예제를 택한다.
- 코드 블록 표시는 쓰지 말고, code 필드에 순수 코드만 넣어라.
- 설치 명령은 'pip install ...' 형태로만 넣어라 (여러 줄 가능).
- README에 설치 명령이 없더라도, 코드에 필요한 최소 라이브러리(예: transformers)가 보이면 포함해라.
- torch 설치는 환경 의존성이 크므로 코드에서 실제로 필요할 때만 넣어라.

출력 JSON 스키마:
{
  "pip_installs": ["pip install transformers>=4.46.0", ...],
  "filename": "run_tmp.py",
  "code_language": "python",
  "code": "<여기에 파이썬 코드 원문>",
  "notes": "선택. 간단한 비고"
}
"""

def _ask_plan_from_readme(readme: str) -> Dict[str, Any]:
    resp = _client.chat.completions.create(
        model=OPENAI_MODEL,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": readme},
        ],
        temperature=0  # 일부 모델이 미지원이면 제거해도 됨
    )
    try:
        return json.loads(resp.choices[0].message.content)
    except Exception:
        return {}

# ─────────────────────────────────────────
# 설치 명령 정규화 & 실행
# ─────────────────────────────────────────
def _normalize_pip_cmd(cmd: str) -> List[str] | None:
    """
    'pip install ...' / 'pip3 install ...' / 'python -m pip install ...' 등을
    현재 파이썬으로 실행 가능한 명령 배열로 표준화.
    """
    if not isinstance(cmd, str):
        return None
    cmd = cmd.strip().lstrip("!").replace("pip3", "pip")
    m = re.search(r"(?:python\s*-m\s+)?pip\s+install\s+(.+)", cmd, flags=re.I)
    if not m:
        return None
    args = shlex.split(m.group(1))
    return [sys.executable, "-m", "pip", "install", "--disable-pip-version-check"] + args

def _ensure_minimal_installs(plan: Dict[str, Any]) -> List[str]:
    """
    README에 설치 명령이 비어있거나 부족할 때 최소 보정.
    code 내용에서 import 흔적을 보고 transformers/torch 추가.
    """
    installs = [c for c in (plan.get("pip_installs") or []) if isinstance(c, str)]
    code = plan.get("code") or ""
    if "transformers" in code and not any("transformers" in c for c in installs):
        installs.append("pip install transformers>=4.46.0")
    if re.search(r"\bimport\s+torch\b|\btorch\.", code) and not any(c.strip().startswith("pip install torch") for c in installs):
        installs.append("pip install torch")
    return installs

def _run_cmd(cmd_list: List[str], cwd: Path | None = None, timeout: int | None = None) -> Dict[str, Any]:
    try:
        proc = subprocess.run(
            cmd_list,
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            encoding=ENC_RUN,
            timeout=timeout
        )
        return {
            "cmd": " ".join(cmd_list),
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr
        }
    except subprocess.TimeoutExpired as e:
        return {
            "cmd": " ".join(cmd_list),
            "returncode": -9,
            "stdout": e.stdout or "",
            "stderr": f"TimeoutExpired: {e}"
        }
    except Exception as e:
        return {
            "cmd": " ".join(cmd_list),
            "returncode": -1,
            "stdout": "",
            "stderr": f"{type(e).__name__}: {e}"
        }

# ─────────────────────────────────────────
# 안전한 출력 폴더 결정
# ─────────────────────────────────────────
def _safe_outdir(output_dir: str | Path | None) -> Path:
    """
    - 명시된 output_dir이 없으면 환경변수 기본(_DEFAULT_OUTDIR)
    - 실수로 긴 문장/프롬프트가 넘어오면 무시하고 기본 폴더 사용
    """
    if output_dir is None:
        return Path(_DEFAULT_OUTDIR)

    p = Path(output_dir)
    name = p.name
    # 공백 과다/특수문자 많으면 프롬프트로 오인 → 기본 폴더 사용
    if len(name) > 48 or re.search(r"[\\/:*?\"<>|]", name) or len(re.findall(r"\s", name)) > 6:
        return Path(_DEFAULT_OUTDIR)
    return p

# ─────────────────────────────────────────
# 메인: 계획 추출 → 설치 → 코드 실행 → 결과 JSON 저장
# ─────────────────────────────────────────
def run_inference(readme: str, output_dir: str | Path | None = None) -> Path:
    outdir = _safe_outdir(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) GPT에게 실행 계획 요청
    plan = _ask_plan_from_readme(readme)
    if not plan or not isinstance(plan, dict):
        plan = {}

    # 필수 필드 기본값 보정
    filename = plan.get("filename") or "run_tmp.py"
    code = (plan.get("code") or "").strip()
    plan["filename"] = filename
    plan.setdefault("code_language", "python")
    plan.setdefault("pip_installs", [])
    plan.setdefault("notes", "")

    # 2) 설치 명령 보정/추가
    plan["pip_installs"] = _ensure_minimal_installs(plan)

    # 3) 계획 JSON 저장 (모델 폴더)
    plan_path = outdir / "inference_plan.json"
    with open(plan_path, "w", encoding="utf-8") as f:
        json.dump(plan, f, ensure_ascii=False, indent=2)

    install_logs: List[Dict[str, Any]] = []

    # 4) 의존성 설치 실행 (작업 디렉토리 = 모델 폴더)
    for raw in plan["pip_installs"]:
        norm = _normalize_pip_cmd(raw)
        if not norm:
            install_logs.append({
                "cmd": raw, "returncode": -1, "stdout": "", "stderr": "Unrecognized pip command"
            })
            continue
        print(f"📦 Installing: {' '.join(norm)}")
        log = _run_cmd(norm, cwd=outdir, timeout=1200)  # 최대 20분
        install_logs.append(log)

    # 5) 코드 파일 생성 (모델 폴더)
    codefile = outdir / filename
    created_code = False
    if code:
        with open(codefile, "w", encoding="utf-8") as f:
            f.write(code)
        created_code = True
        print(f"✅ 3. 추출된 코드를 '{codefile.name}' 파일로 성공적으로 저장했습니다.")
    else:
        print("⚠️ 실행 코드가 비어 있습니다. (README에 로컬 실행 예제가 없을 수 있음)")

    # 6) 코드 실행 (작업 디렉토리 = 모델 폴더)
    exec_log = {"cmd": "", "returncode": None, "stdout": "", "stderr": ""}
    if created_code:
        print(f"\n✅ 4. '{codefile.name}' 스크립트 실행을 시작합니다...")
        print("   (모델 다운로드 및 실행에 시간이 걸릴 수 있습니다.)\n")
        exec_cmd = [sys.executable, filename]
        exec_log = _run_cmd(exec_cmd, cwd=outdir, timeout=1800)  # 최대 30분
        # 실행 후 임시 파일 제거
        try:
            codefile.unlink(missing_ok=True)
            print(f"\n🧹 '{codefile.name}' 임시 파일을 삭제했습니다.")
        except Exception:
            pass

    # 7) 결과 JSON 저장 (모델 폴더)
    result = {
        "plan_path": str(plan_path),
        "plan": plan,
        "installs": install_logs,
        "execution": exec_log,
    }
    result_path = outdir / "inference_result.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # 콘솔 요약
    print("\n" + "-"*60)
    print(f"📄 계획 JSON: {plan_path}")
    print(f"📄 결과 JSON: {result_path}")
    print(f"▶ 실행 반환코드: {exec_log.get('returncode')}")
    print("-"*60)

    return result_path

# ─────────────────────────────────────────
# 단독 실행 테스트
# ─────────────────────────────────────────


if __name__ == "__main__":
    # 데모: README 문자열을 받아서 실행
    demo_readme = "여기에 README 텍스트를 넣으세요."
    run_inference(demo_readme, output_dir=".")


# if __name__=="__main__":
#     readme = """
#     ---\nlicense: apache-2.0\nlicense_link: https://huggingface.co/skt/A.X-4.0-Light/blob/main/LICENSE\nlanguage:\n- en\n- ko\npipeline_tag: text-generation\nlibrary_name: transformers\nmodel_id: skt/A.X-4.0-Light\ndevelopers: SKT AI Model Lab\nmodel-index:\n- name: A.X-4.0-Light\n  results:\n  - task:\n      type: generate_until\n      name: mmlu\n    dataset:\n      name: mmlu (chat CoT)\n      type: hails/mmlu_no_train\n    metrics:\n    - type: exact_match\n      value: 75.43\n      name: exact_match\n  - task:\n      type: generate_until\n      name: kmmlu\n    dataset:\n      name: kmmlu (chat CoT)\n      type: HAERAE-HUB/KMMLU\n    metrics:\n    - type: exact_match\n      value: 64.15\n      name: exact_match\n---\n\n# A.X 4.0 Light\n\n<p align=\"center\">\n    <picture>\n        <img src=\"./assets/A.X_logo_ko_4x3.png\" width=\"45%\" style=\"margin: 40px auto;\">\n    </picture>\n</p>\n<p align=\"center\"> <a href=\"https://huggingface.co/collections/skt/ax-4-68637ebaa63b9cc51925e886\">🤗 Models</a>   |   <a href=\"https://sktax.chat/chat\">💬 Chat</a>   |    <a href=\"https://github.com/SKT-AI/A.X-4.0/blob/main/apis/README.md\">📬 APIs (FREE!)</a>    |   <a href=\"https://github.com/SKT-AI/A.X-4.0\">🖥️ Github</a> </p>\n\n## A.X 4.0 Family Highlights\n\nSK Telecom released **A.X 4.0** (pronounced \"A dot X\"), a large language model (LLM) optimized for Korean-language understanding and enterprise deployment, on July 03, 2025. Built on the open-source [Qwen2.5](https://huggingface.co/collections/Qwen/qwen25-66e81a666513e518adb90d9e) model, A.X 4.0 has been further trained with large-scale Korean datasets to deliver outstanding performance in real-world business environments.\n\n- **Superior Korean Proficiency**: Achieved a score of 78.3 on [KMMLU](https://huggingface.co/datasets/HAERAE-HUB/KMMLU), the leading benchmark for Korean-language evaluation and a Korean-specific adaptation of MMLU, outperforming GPT-4o (72.5).\n- **Deep Cultural Understanding**: Scored 83.5 on [CLIcK](https://huggingface.co/datasets/EunsuKim/CLIcK), a benchmark for Korean cultural and contextual comprehension, surpassing GPT-4o (80.2).\n- **Efficient Token Usage**: A.X 4.0 uses approximately 33% fewer tokens than GPT-4o for the same Korean input, enabling more cost-effective and efficient processing.\n- **Deployment Flexibility**: Offered in both a 72B-parameter standard model (A.X 4.0) and a 7B lightweight version (A.X 4.0 Light).\n- **Long Context Handling**: Supports up to 131,072 tokens, allowing comprehension of lengthy documents and conversations. (Lightweight model supports up to 16,384 tokens length)\n\n## Performance\n\n### Model Performance\n\n<table><thead>\n  <tr>\n    <th colspan=\"2\">Benchmarks</th>\n    <th>A.X 4.0</th>\n    <th>Qwen3-235B-A22B<br/>(w/o reasoning)</th>\n    <th>Qwen2.5-72B</th>\n    <th>GPT-4o</th>\n  </tr></thead>\n<tbody>\n  <tr>\n    <td rowspan=\"6\">Knowledge</td>\n    <td>KMMLU</td>\n    <td>78.32</td>\n    <td>73.64</td>\n    <td>66.44</td>\n    <td>72.51</td>\n  </tr>\n  <tr>\n    <td>KMMLU-pro</td>\n    <td>72.43</td>\n    <td>64.4</td>\n    <td>56.27</td>\n    <td>66.97</td>\n  </tr>\n  <tr>\n    <td>KMMLU-redux</td>\n    <td>74.18</td>\n    <td>71.17</td>\n    <td>58.76</td>\n    <td>69.08</td>\n  </tr>\n  <tr>\n    <td>CLIcK</td>\n    <td>83.51</td>\n    <td>74.55</td>\n    <td>72.59</td>\n    <td>80.22</td>\n  </tr>\n  <tr>\n    <td>KoBALT</td>\n    <td>47.30</td>\n    <td>41.57</td>\n    <td>37.00</td>\n    <td>44.00</td>\n  </tr>\n  <tr>\n    <td>MMLU</td>\n    <td>86.62</td>\n    <td>87.37</td>\n    <td>85.70</td>\n    <td>88.70</td>\n  </tr>\n  <tr>\n    <td rowspan=\"3\">General</td>\n    <td>Ko-MT-Bench</td>\n    <td>86.69</td>\n    <td>88.00</td>\n    <td>82.69</td>\n    <td>88.44</td>\n  </tr>\n  <tr>\n    <td>MT-Bench</td>\n    <td>83.25</td>\n    <td>86.56</td>\n    <td>93.50</td>\n    <td>88.19</td>\n  </tr>\n  <tr>\n    <td>LiveBench<sup>2024.11</sup></td>\n    <td>52.30</td>\n    <td>64.50</td>\n    <td>54.20</td>\n    <td>52.19</td>\n  </tr>\n  <tr>\n    <td rowspan=\"2\">Instruction Following</td>\n    <td>Ko-IFEval</td>\n    <td>77.96</td>\n    <td>77.53</td>\n    <td>77.07</td>\n    <td>75.38</td>\n  </tr>\n  <tr>\n    <td>IFEval</td>\n    <td>86.05</td>\n    <td>85.77</td>\n    <td>86.54</td>\n    <td>83.86</td>\n  </tr>\n  <tr>\n    <td rowspan=\"2\">Math</td>\n    <td>HRM8K</td>\n    <td>48.55</td>\n    <td>54.52</td>\n    <td>46.37</td>\n    <td>43.27</td>\n  </tr>\n  <tr>\n    <td>MATH</td>\n    <td>74.28</td>\n    <td>72.72</td>\n    <td>77.00</td>\n    <td>72.38</td>\n  </tr>\n  <tr>\n    <td rowspan=\"3\">Code</td>\n    <td>HumanEval+</td>\n    <td>79.27</td>\n    <td>79.27</td>\n    <td>81.71</td>\n    <td>86.00</td>\n  </tr>\n  <tr>\n    <td>MBPP+</td>\n    <td>73.28</td>\n    <td>70.11</td>\n    <td>75.66</td>\n    <td>75.10</td>\n  </tr>\n  <tr>\n    <td>LiveCodeBench<sup>2024.10~2025.04</sup></td>\n    <td>26.07</td>\n    <td>33.09</td>\n    <td>27.58</td>\n    <td>29.30</td>\n  </tr>\n  <tr>\n    <td>Long Context</td>\n    <td>LongBench<sup>&lt;128K</sup></td>\n    <td>56.70</td>\n    <td>49.40</td>\n    <td>45.60</td>\n    <td>47.50</td>\n  </tr>\n  <tr>\n    <td>Tool-use</td>\n    <td>FunctionChatBench</td>\n    <td>85.96</td>\n    <td>82.43</td>\n    <td>88.30</td>\n    <td>95.70</td>\n  </tr>\n</tbody></table>\n\n### Lightweight Model Performance\n\n<table><thead>\n  <tr>\n    <th colspan=\"2\">Benchmarks</th>\n    <th>A.X 4.0 Light</th>\n    <th>Qwen3-8B<br/>(w/o reasoning)</th>\n    <th>Qwen2.5-7B</th>\n    <th>EXAONE-3.5-7.8B</th>\n    <th>Kanana-1.5-8B</th>\n  </tr></thead>\n<tbody>\n  <tr>\n    <td rowspan=\"6\">Knowledge</td>\n    <td>KMMLU</td>\n    <td>64.15</td>\n    <td>63.53</td>\n    <td>49.56</td>\n    <td>53.76</td>\n    <td>48.28</td>\n  </tr>\n  <tr>\n    <td>KMMLU-pro</td>\n    <td>50.28</td>\n    <td>50.71</td>\n    <td>38.87</td>\n    <td>40.11</td>\n    <td>37.63</td>\n  </tr>\n  <tr>\n    <td>KMMLU-redux</td>\n    <td>56.05</td>\n    <td>55.74</td>\n    <td>38.58</td>\n    <td>42.21</td>\n    <td>35.33</td>\n  </tr>\n  <tr>\n    <td>CLIcK</td>\n    <td>68.05</td>\n    <td>62.71</td>\n    <td>60.56</td>\n    <td>64.30</td>\n    <td>61.30</td>\n  </tr>\n  <tr>\n    <td>KoBALT</td>\n    <td>30.29</td>\n    <td>26.57</td>\n    <td>21.57</td>\n    <td>21.71</td>\n    <td>23.14</td>\n  </tr>\n  <tr>\n    <td>MMLU</td>\n    <td>75.43</td>\n    <td>82.89</td>\n    <td>75.40</td>\n    <td>72.20</td>\n    <td>68.82</td>\n  </tr>\n  <tr>\n    <td rowspan=\"3\">General</td>\n    <td>Ko-MT-Bench</td>\n    <td>79.50</td>\n    <td>64.06</td>\n    <td>61.31</td>\n    <td>81.06</td>\n    <td>76.30</td>\n  </tr>\n  <tr>\n    <td>MT-Bench</td>\n    <td>81.56</td>\n    <td>65.69</td>\n    <td>79.37</td>\n    <td>83.50</td>\n    <td>77.60</td>\n  </tr>\n  <tr>\n    <td>LiveBench</td>\n    <td>37.10</td>\n    <td>50.20</td>\n    <td>37.00</td>\n    <td>40.20</td>\n    <td>29.40</td>\n  </tr>\n  <tr>\n    <td rowspan=\"2\">Instruction Following</td>\n    <td>Ko-IFEval</td>\n    <td>72.99</td>\n    <td>73.39</td>\n    <td>60.73</td>\n    <td>65.01</td>\n    <td>69.96</td>\n  </tr>\n  <tr>\n    <td>IFEval</td>\n    <td>84.68</td>\n    <td>85.38</td>\n    <td>76.73</td>\n    <td>82.61</td>\n    <td>80.11</td>\n  </tr>\n  <tr>\n    <td rowspan=\"2\">Math</td>\n    <td>HRM8K</td>\n    <td>40.12</td>\n    <td>52.50</td>\n    <td>35.13</td>\n    <td>31.88</td>\n    <td>30.87</td>\n  </tr>\n  <tr>\n    <td>MATH</td>\n    <td>68.88</td>\n    <td>71.48</td>\n    <td>65.58</td>\n    <td>63.20</td>\n    <td>59.28</td>\n  </tr>\n  <tr>\n    <td rowspan=\"3\">Code</td>\n    <td>HumanEval+</td>\n    <td>75.61</td>\n    <td>77.44</td>\n    <td>74.39</td>\n    <td>76.83</td>\n    <td>76.83</td>\n  </tr>\n  <tr>\n    <td>MBPP+</td>\n    <td>67.20</td>\n    <td>62.17</td>\n    <td>68.50</td>\n    <td>64.29</td>\n    <td>67.99</td>\n  </tr>\n  <tr>\n    <td>LiveCodeBench</td>\n    <td>18.03</td>\n    <td>23.93</td>\n    <td>16.62</td>\n    <td>17.98</td>\n    <td>16.52</td>\n  </tr>\n</tbody></table>\n\n## 🚀 Quickstart\n\n### with HuggingFace Transformers\n\n- `transformers>=4.46.0` or the latest version is required to use `skt/A.X-4.0-Light`\n```bash\npip install transformers>=4.46.0\n```\n\n#### Example Usage\n\n```python\nimport torch\nfrom transformers import AutoModelForCausalLM, AutoTokenizer\n\nmodel_name = \"skt/A.X-4.0-Light\"\nmodel = AutoModelForCausalLM.from_pretrained(\n    model_name,\n    torch_dtype=torch.bfloat16,\n    device_map=\"auto\",\n)\nmodel.eval()\ntokenizer = AutoTokenizer.from_pretrained(model_name)\n\nmessages = [\n    {\"role\": \"system\", \"content\": \"당신은 사용자가 제공하는 영어 문장들을 한국어로 번역하는 AI 전문가입니다.\"},\n    {\"role\": \"user\", \"content\": \"The first human went into space and orbited the Earth on April 12, 1961.\"},\n]\ninput_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors=\"pt\").to(model.device)\n\nwith torch.no_grad():\n    output = model.generate(\n        input_ids,\n        max_new_tokens=128,\n        do_sample=False,\n    )\n\nlen_input_prompt = len(input_ids[0])\nresponse = tokenizer.decode(output[0][len_input_prompt:], skip_special_tokens=True)\nprint(response)\n# Output:\n# 1961년 4월 12일, 최초의 인간이 우주로 나가 지구를 공전했습니다.\n```\n\n### with vLLM\n\n- `vllm>=v0.6.4.post1` or the latest version is required to use tool-use function\n```bash\npip install vllm>=v0.6.4.post1\n# if you don't want to activate tool-use function, just commenting out below vLLM option\nVLLM_OPTION=\"--enable-auto-tool-choice --tool-call-parser hermes\"\nvllm serve skt/A.X-4.0-Light $VLLM_OPTION\n```\n\n#### Example Usage \n  \n```python\nfrom openai import OpenAI\n\ndef call(messages, model):\n    completion = client.chat.completions.create(\n        model=model,\n        messages=messages,\n    )\n    print(completion.choices[0].message)\n\nclient = OpenAI(\n    base_url=\"http://localhost:8000/v1\",\n    api_key=\"api_key\"\n)\nmodel = \"skt/A.X-4.0-Light\"\nmessages = [{\"role\": \"user\", \"content\": \"에어컨 여름철 적정 온도는? 한줄로 답변해줘\"}]\ncall(messages, model)\n# Output:\n# ChatCompletionMessage(content='여름철 적정 에어컨 온도는 일반적으로 24-26도입니다.', refusal=None, role='assistant', audio=None, function_call=None, tool_calls=[], reasoning_content=None)\n\nmessages = [{\"role\": \"user\", \"content\": \"What is the appropriate temperature for air conditioning in summer? Response in a single sentence.\"}]\ncall(messages, model)\n# Output:\n# ChatCompletionMessage(content='The appropriate temperature for air conditioning in summer generally ranges from 72°F to 78°F (22°C to 26°C) for comfort and energy efficiency.', refusal=None, role='assistant', audio=None, function_call=None, tool_calls=[], reasoning_content=None)\n```\n\n#### Examples for tool-use\n```python\nfrom openai import OpenAI\n\n\ndef call(messages, model):\n    completion = client.chat.completions.create(\n        model=model,\n        messages=messages,\n        tools=tools\n    )\n    print(completion.choices[0].message)\n\n\nclient = OpenAI(\n    base_url=\"http://localhost:8000/v1\",\n    api_key=\"api_key\"\n)\nmodel = \"skt/A.X-4.0-Light\"\n\ncalculate_discount = {\n    \"type\": \"function\",\n    \"function\": {\n        \"name\": \"calculate_discount\",\n        \"description\": \"원가격과 할인율(퍼센트 단위)을 입력받아 할인된 가격을계산한다.\",\n        \"parameters\": {\n            \"type\": \"object\",\n            \"properties\": {\n                \"original_price\": {\n                    \"type\": \"number\",\n                    \"description\": \"상품의 원래 가격\"\n                },\n                \"discount_percentage\": {\n                    \"type\": \"number\",\n                    \"description\": \"적용할 할인율(예: 20% 할인의 경우 20을 입력)\"\n                }\n            },\n            \"required\": [\"original_price\", \"discount_percentage\"]\n        }\n    }\n}\nget_exchange_rate = {\n    \"type\": \"function\",\n    \"function\": {\n        \"name\": \"get_exchange_rate\",\n        \"description\": \"두 통화 간의 환율을 가져온다.\",\n        \"parameters\": {\n            \"type\": \"object\",\n            \"properties\": {\n                \"base_currency\": {\n                    \"type\": \"string\",\n                    \"description\": \"The currency to convert from.\"\n                },\n                \"target_currency\": {\n                    \"type\": \"string\",\n                    \"description\": \"The currency to convert to.\"\n                }\n            },\n            \"required\": [\"base_currency\", \"target_currency\"]\n        }\n    }\n}\ntools = [calculate_discount, get_exchange_rate]\n\n### Slot filling ###\nmessages = [{\"role\": \"user\", \"content\": \"우리가 뭘 사야되는데 원래 57600원인데 직원할인 받을 수 있거든? 할인가좀 계산해줘\"}]\ncall(messages, model)\n# Output:\n# ChatCompletionMessage(content='할인율을 알려주시겠습니까?', refusal=None, role='assistant', audio=None, function_call=None, tool_calls=[], reasoning_content=None)\n\n\n### Function calling ###\nmessages = [\n    {\"role\": \"user\", \"content\": \"우리가 뭘 사야되는데 원래 57600원인데 직원할인 받을 수 있거든? 할인가좀 계산해줘\"},\n    {\"role\": \"assistant\", \"content\": \"할인율을 알려주시겠습니까?\"},\n    {\"role\": \"user\", \"content\": \"15% 할인 받을 수 있어.\"},\n]\ncall(messages, model)\n# Output: \n# ChatCompletionMessage(content=None, refusal=None, role='assistant', audio=None, function_call=None, tool_calls=[ChatCompletionMessageToolCall(id='chatcmpl-tool-7778d1d9fca94bf2acbb44c79359502c', function=Function(arguments='{\"original_price\": 57600, \"discount_percentage\": 15}', name='calculate_discount'), type='function')], reasoning_content=None)\n\n\n### Completion ###\nmessages = [\n    {\"role\": \"user\", \"content\": \"우리가 뭘 사야되는데 원래 57600원인데 직원할인 받을 수 있거든? 할인가좀 계산해줘\"},\n    {\"role\": \"assistant\", \"content\": \"할인율을 알려주시겠습니까?\"},\n    {\"role\": \"user\", \"content\": \"15% 할인 받을 수 있어.\"},\n    {\"role\": \"tool\", \"tool_call_id\": \"random_id\", \"name\": \"calculate_discount\", \"content\": \"{\\\"original_price\\\": 57600, \\\"discount_percentage\\\": 15, \\\"discounted_price\\\": 48960.0}\"}\n]\ncall(messages, model)\n# Output: \n# ChatCompletionMessage(content='57600원의 상품에서 15% 할인을 적용하면, 할인된 가격은 48960원입니다.', refusal=None, role='assistant', audio=None, function_call=None, tool_calls=[], reasoning_content=None)\n```\n\n## License\n\nThe `A.X 4.0 Light` model is licensed under `Apache License 2.0`.\n\n## Citation\n```\n@article{SKTAdotX4Light,\n  title={A.X 4.0 Light},\n  author={SKT AI Model Lab},\n  year={2025},\n  url={https://huggingface.co/skt/A.X-4.0-Light}\n}\n```\n\n## Contact\n\n- Business & Partnership Contact: [a.x@sk.com](a.x@sk.com)
#     """
#     run_inference(readme)