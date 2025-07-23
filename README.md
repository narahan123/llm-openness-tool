# 내 프로젝트 제목

프로젝트에 대한 설명입니다.

## 시스템 아키텍처

다음은 시스템의 구성 요소와 데이터 흐름을 시각화한 다이어그램입니다.

```mermaid
graph TD
    A["사용자 입력\n(URL 또는 org/model)"] --> B["모델 식별\n(model_Identifier.py)"]
    B --> C1["Hugging Face 수집\n(huggingface_Fatcher.py)"]
    C1 --> C2["arXiv 논문 수집\n(※ Hugging Face에 포함된 arXiv 태그 기반)\n(arxiv_Fetcher.py)"]

    C1 --> D1["Hugging Face 필터링\n(huggingface_Dispatcher.py)"]
    C2 --> D2["arXiv 필터링\n(arxiv_Dispatcher.py)"]

    B --> E1["GitHub 수집\n(github_Fatcher.py)"]
    E1 --> E2["GitHub 필터링\n(github_Dispatcher.py)"]

    D1 --> F["개방성 평가\n(openness_Evaluator.py)"]
    D2 --> F
    E2 --> F

    F --> G["개방성 점수 저장"]
    B --> H["모델 추론\n(inference.py)"]
