# 내 프로젝트 제목

프로젝트에 대한 설명입니다.

## 시스템 아키텍처

다음은 시스템의 구성 요소와 데이터 흐름을 시각화한 다이어그램입니다.

```mermaid
graph TD

    %% 사용자 입력
    A[사용자 입력: 모델명 또는 URL]

    %% 입력 처리 모듈
    subgraph InputModule[1. 입력 처리 모듈]
        B1[입력 파싱 및 정규화]
        B2[캐시 확인]
        B3[메타 매핑 조회]
        B4[평가 대상 리소스 후보 생성]
    end

    %% 비동기 메시지 큐
    subgraph Queue[2. 메시지 큐 / 태스크 큐]
        Q[Celery / RabbitMQ]
    end

    %% 리소스 디스커버리
    subgraph Fetcher[3. 리소스 디스커버리 에이전트]
        F1[ArxivFetcher]
        F2[HuggingFaceFetcher]
        F3[GitHubFetcher]
        F4[BlogCrawler]
    end

    %% 파싱 모듈
    subgraph Parser[4. 정보 추출 및 파싱]
        P1[규칙/NLP 기반 파싱]
        P2[추출 결과 (JSON)]
    end

    %% 평가 엔진
    subgraph Evaluator[5. 평가 엔진]
        E1[프레임워크 매핑 및 점수화]
        E2[수동 오버라이드 (UI 연동)]
    end

    %% DB & 캐시
    subgraph DB[6. 데이터 관리 및 캐시]
        D1[PostgreSQL: 평가 결과 저장]
        D2[Redis: API 호출 캐시]
    end

    %% 흐름 연결
    A --> B1 --> B2 --> B3 --> B4 --> Q
    Q --> F1 --> Q
    Q --> F2 --> Q
    Q --> F3 --> Q
    Q --> F4 --> Q
    Q --> P1 --> P2 --> Q
    Q --> E1 --> E2 --> D1
    B2 --> D2
    P2 --> D1
