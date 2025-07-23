# 내 프로젝트 제목

프로젝트에 대한 설명입니다.

## 시스템 아키텍처

다음은 시스템의 구성 요소와 데이터 흐름을 시각화한 다이어그램입니다.

```mermaid
graph TD
    A["사용자 입력\n(URL 또는 org/model)"] --> B["모델 식별\n(model_Identifier.py)"]
    B --> C1["Hugging Face 수집\n(huggingface_Fatcher.py)"]
    B --> C2["GitHub 수집\n(github_Fatcher.py)"]
    B --> C3["arXiv 논문 수집\n(arxiv_Fetcher.py)"]

    C1 --> D1["Hugging Face 필터링\n(huggingface_Dispatcher.py)"]
    C2 --> D2["GitHub 필터링\n(github_Dispatcher.py)"]
    C3 --> D3["arXiv 필터링\n(arxiv_Dispatcher.py)"]

    D1 --> E["개방성 평가\n(openness_Evaluator.py)"]
    D2 --> E
    D3 --> E

    E --> F["개방성 점수 저장"]
    B --> G["모델 추론\n(inference.py)"]
