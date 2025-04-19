# 내 프로젝트 제목

프로젝트에 대한 설명입니다.

## 시스템 아키텍처

다음은 시스템의 구성 요소와 데이터 흐름을 시각화한 다이어그램입니다.

```mermaid
graph TD

    %% 사용자 입력
    user_input[사용자 입력 (모델명 or URL)]

    %% 입력 처리
    subgraph InputProcessor[1. Input Processor]
        parse_model[모델명/URL 파싱 및 식별자 정규화]
    end

    %% FetchRequest 생성 및 병렬 분배
    user_input --> parse_model --> request[FetchRequest 생성] --> dispatcher[플러그인에 병렬 전파]

    %% 리소스 디스커버리
    subgraph ResourceDiscoveryAgent[2. Resource Discovery Agent]
        arxiv[ArxivFetcher]
        hf[HuggingFaceFetcher]
        gh[GitHubFetcher]
        blog[BlogCrawler]
    end

    dispatcher --> arxiv --> raw_arxiv[논문 메타데이터]
    dispatcher --> hf --> raw_hf[모델카드 텍스트]
    dispatcher --> gh --> raw_gh[README / LICENSE]
    dispatcher --> blog --> raw_blog[웹 게시물 / 블로그]

    %% 정보 추출 및 파싱
    subgraph ExtractParse[3. 정보 추출 및 파싱]
        rule_parse[규칙 기반 파싱 (정규표현식, 태그)]
        nlp_parse[경량 NLP 파이프라인 (의존성 해석)]
        merge[16개 평가항목 관련 정보 추출]
    end

    raw_arxiv --> rule_parse
    raw_hf --> rule_parse
    raw_gh --> rule_parse
    raw_blog --> rule_parse
    rule_parse --> nlp_parse --> merge

    %% 평가 엔진
    subgraph EvaluationEngine[4. 평가 엔진]
        eval_model[텍스트 분류 모델 기반 평가]
        score_map[개방성 점수 계산 (Open:1, Semi:0.5, Closed:0)]
        output[최종 리더보드 및 시각화]
    end

    merge --> eval_model --> score_map --> output


