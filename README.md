# 내 프로젝트 제목

프로젝트에 대한 설명입니다.

## 시스템 아키텍처

다음은 시스템의 구성 요소와 데이터 흐름을 시각화한 다이어그램입니다.

```mermaid
graph TD
    A[사용자 입력<br>모델명/URL] -->|모델 식별자 추출| B(입력 프로세스<br>모델 식별자 정규화<br>Fetchrequest 생성)
    
    B -->|Fetchrequest 병렬 전파| C{ArxivFetcher<br>arXiv API}
    B -->|Fetchrequest 병렬 전파| D{HuggingFaceFetcher<br>Hub API}
    B -->|Fetchrequest 병렬 전파| E{GitHubFetcher<br>GitHub API}
    B -->|Fetchrequest 병렬 전파| F{BlogCrawler<br>웹 크롤러}
    
    C -->|정보 전달| G[정보 수집 프로세스<br>중앙 큐]
    D -->|정보 전달| G
    E -->|정보 전달| G
    F -->|정보 전달| G
    
    G -->|경량 NLP 모델<br>의존 구문 분석<br>키워드 스팟팅| H[평가 항목 정보 추출<br>16개 항목 문장 해석]
    
    H -->|추출 정보 전달| I[평가 에이전트<br>코드 구동/재현 판단<br>텍스트 분류 모델<br>개방성 점수 도출]
    
    I -->|모델 정보 저장| J[리더보드 생성<br>데이터베이스 저장<br>개방성 점수/항목 비교<br>순위 도출]

    style A fill:#f5f5f5,stroke:#333,stroke-width:1px
    style B fill:#f5f5f5,stroke:#333,stroke-width:1px
    style C fill:#f5f5f5,stroke:#333,stroke-width:1px
    style D fill:#f5f5f5,stroke:#333,stroke-width:1px
    style E fill:#f5f5f5,stroke:#333,stroke-width:1px
    style F fill:#f5f5f5,stroke:#333,stroke-width:1px
    style G fill:#f5f5f5,stroke:#333,stroke-width:1px
    style H fill:#f5f5f5,stroke:#333,stroke-width:1px
    style I fill:#f5f5f5,stroke:#333,stroke-width:1px
    style J fill:#f5f5f5,stroke:#333,stroke-width:1px

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
    B --> G[모델 추론<br/>(inference.py)
