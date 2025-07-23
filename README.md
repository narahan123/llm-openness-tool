# 내 프로젝트 제목

프로젝트에 대한 설명입니다.

## 시스템 아키텍처

다음은 시스템의 구성 요소와 데이터 흐름을 시각화한 다이어그램입니다.

```mermaid
graph TD
    A[사용자 입력<br/>URL or org/model] --> B[모델 식별<br/>(model_Identifier.py)]
    B --> C1[Hugging Face 정보 수집<br/>(huggingface_Fatcher.py)]
    B --> C2[GitHub 정보 수집<br/>(github_Fatcher.py)]
    B --> C3[arXiv 논문 수집<br/>(arxiv_Fetcher.py)]

    C1 --> D1[Hugging Face 필터링<br/>(huggingface_Dispatcher.py)]
    C2 --> D2[GitHub 필터링<br/>(github_Dispatcher.py)]
    C3 --> D3[arXiv 필터링<br/>(arxiv_Dispatcher.py)]

    D1 --> E[openness_Evaluator.py]
    D2 --> E
    D3 --> E

    E --> F[개방성 점수 JSON 저장]
    B --> G[모델 추론<br/>(inference.py)]
