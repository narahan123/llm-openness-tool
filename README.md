# 내 프로젝트 제목

프로젝트에 대한 설명입니다.

## 시스템 아키텍처

다음은 시스템의 구성 요소와 데이터 흐름을 시각화한 다이어그램입니다.

```mermaid
graph TD
    subgraph "사용자 인터페이스"
        A[사용자 입력 (모델명 / URL)]
    end

    subgraph "입력 처리 모듈"
        B1[입력 파싱 및 정규화]
        B2[캐시 확인]
        B3[메타 매핑 조회]
        B4[평가 대상 리소스 후보 생성]
    end

    subgraph "비동기 워크플로우"
        C[메시지 큐 / 태스크 큐 (RabbitMQ / Celery)]
    end

    subgraph "리소스 디스커버리 에이전트"
        D1[ArxivFetcher]
        D2[HuggingFaceFetcher]
        D3[GitHubFetcher]
        D4[BlogCrawler]
    end

    subgraph "정보 추출 및 파싱"
        E1[규칙/NLP 기반 처리]
        E2[추출 결과 (JSON)]
    end

    subgraph "평가 엔진"
        F1[프레임워크 매핑 및 점수화]
        F2[수동 오버라이드 (UI 연동)]
    end

    subgraph "데이터 관리 및 캐시"
        G1[PostgreSQL: 모델 메타, 평가 결과]
        G2[Redis: API 호출 캐시]
    end

    %% 흐름 연결
    A --> B1 --> B2 --> B3 --> B4 --> C
    C --> D1 --> C
    C --> D2 --> C
    C --> D3 --> C
    C --> D4 --> C
    C --> E1 --> E2 --> C
    C --> F1 --> F2 --> G1
    E2 --> G1
    B2 --> G2
