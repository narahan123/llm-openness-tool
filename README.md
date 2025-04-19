# 내 프로젝트 제목

프로젝트에 대한 설명입니다.

## 시스템 아키텍처

다음은 시스템의 구성 요소와 데이터 흐름을 시각화한 다이어그램입니다.

```mermaid
graph TD

    %% 사용자 입력
    input_user[사용자 입력: 모델명 또는 URL]

    %% 입력 처리 모듈
    subgraph InputModule[1. 입력 처리 모듈]
        parse[입력 파싱 및 정규화]
        cache[캐시 확인]
        mapping[메타 매핑 조회]
        candidate[평가 대상 리소스 후보 생성]
    end

    %% 비동기 메시지 큐
    subgraph MessageQueue[2. 메시지 큐 / 태스크 큐]
        queue[Celery / RabbitMQ]
    end

    %% 리소스 디스커버리
    subgraph Fetchers[3. 리소스 디스커버리 에이전트]
        arxiv[ArxivFetcher]
        hf[HuggingFaceFetcher]
        github[GitHubFetcher]
        blog[BlogCrawler]
    end

    %% 파싱 모듈
    subgraph Parser[4. 정보 추출 및 파싱]
        parse_logic[규칙/NLP 기반 파싱]
        result[추출 결과: JSON]
    end

    %% 평가 엔진
    subgraph Evaluator[5. 평가 엔진]
        score[프레임워크 매핑 및 점수화]
        override[수동 오버라이드 - UI 연동]
    end

    %% 데이터 관리
    subgraph DB[6. 데이터 관리 및 캐시]
        pg[PostgreSQL: 평가 결과 저장]
        redis[Redis: API 호출 캐시]
    end

    %% 흐름 연결
    input_user --> parse --> cache --> mapping --> candidate --> queue
    queue --> arxiv --> queue
    queue --> hf --> queue
    queue --> github --> queue
    queue --> blog --> queue
    queue --> parse_logic --> result --> queue
    queue --> score --> override --> pg
    cache --> redis
    result --> pg



graph TD
    subgraph "사용자 인터페이스/입력"
        A["사용자 입력 (모델명/URL)"] 
    end

    subgraph "데이터 관리 및 캐시"
        DB[(PostgreSQL: 모델 메타, 평가 결과)]
        Cache[(Redis: API 호출 캐시)]
    end

    subgraph "비동기 워크플로우 (RabbitMQ/Celery)"
        MQ[메시지 큐 / 태스크 큐]
    end

    subgraph "외부 리소스"
        ExtArxiv[arXiv API]
        ExtHF[Hugging Face]
        ExtGH[GitHub]
        ExtWeb[Web/Blogs]
    end

    %% 1. 입력 처리
    A --> Mod1(1. 입력 처리 모듈);
    Mod1 -- 입력 파싱 & 정규화 --> Mod1;
    Mod1 -- 캐시 확인 --> Cache;
    Mod1 -- 메타 매핑 조회 --> DB;
    Mod1 -- 평가 대상 리소스 후보 --> MQ;

    %% 2. 리소스 디스커버리 (비동기 Fetcher)
    MQ -- Fetch 태스크 할당 --> Agent2{2. 리소스 디스커버리 에이전트};
    subgraph Agent2 ["2. 리소스 디스커버리 에이전트 (Fetcher 플러그인)"]
        direction LR
        F1(ArxivFetcher) --- ExtArxiv;
        F2(HuggingFaceFetcher) --- ExtHF;
        F3(GitHubFetcher) --- ExtGH;
        F4(BlogCrawler) --- ExtWeb;
    end
    Agent2 -- 수집된 원시 데이터 --> MQ;

    %% 3. 정보 추출 및 파싱
    MQ -- 파싱 태스크 할당 --> Mod3(3. 정보 추출·파싱 모듈);
    Mod3 -- 규칙/NLP 기반 처리 --> Mod3;
    Mod3 -- 추출 결과 (JSON) --> MQ;

    %% 4. 평가 엔진
    MQ -- 평가 태스크 할당 --> Mod4(4. 평가 엔진);
    Mod4 -- 프레임워크 매핑 & 점수화 --> Mod4;
    Mod4 -- 평가 결과 저장/조회 --> DB;
    Mod4 -- 수동 오버라이드 (UI 연동) --> Mod4;
    Mod4 --> Output[최종 평가 결과];

    %% 스타일링 (선택 사항)
    classDef module fill:#lightblue,stroke:#333,stroke-width:2px;
    classDef db fill:#lightgreen,stroke:#333,stroke-width:2px;
    classDef queue fill:#orange,stroke:#333,stroke-width:2px;
    classDef external fill:#lightgrey,stroke:#333,stroke-width:1px;
    class A,Output fill:#white,stroke:#333,stroke-width:2px;

    class Mod1,Agent2,Mod3,Mod4 module;
    class DB,Cache db;
    class MQ queue;
    class ExtArxiv,ExtHF,ExtGH,ExtWeb external;


