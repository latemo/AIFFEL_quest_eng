last modified date : 2026.05
제작 : 모두의연구소

# Day 2 실습 — Advanced·Modular RAG + RAGAS 평가

> 수정 버전 메모
> 이 노트북은 Step 6 / Step J가 너무 오래 걸리는 문제를 줄이기 위해 기본 평가 샘플을 20개에서 3개로 낮추고, 진행률 출력과 빠른 rerank 후보 수 설정을 추가한 버전입니다.
> 최종 제출용 비교가 필요하면 `FAST_MODE = False`, `FAST_MODE_KLUE = False`, `RERANK_CANDIDATE_K = 10`, `RERANK_CANDIDATE_K_KLUE = 10` 으로 바꾸면 됩니다.

# 들어가며

Day 1에서는 가장 기본형인 **Naive RAG** 파이프라인을 직접 구현해 보았습니다. 이번 실습에서는 한국어 QA 벤치마크 **KorQuAD v1** 데이터셋 위에서 **Advanced·Modular RAG** 의 핵심 기법(Multi-Query, RAG-Fusion, HyDE, Reranking, Self-RAG)을 단계적으로 적용하고, 그 결과를 **RAGAS** 로 정량 평가합니다.

이번 실습이 끝나면 다음을 직접 말할 수 있게 됩니다.
- Naive RAG 대비 **어떤 단계**를 보강하면 정답률이 올라가는가
- Multi-Query / RAG-Fusion / HyDE / Reranker / Self-RAG 는 각각 **어떤 코드 라인**으로 적용하는가
- RAGAS 의 4대 지표(Faithfulness · Answer Relevance · Context Precision · Context Recall)는 어떻게 계산되고 어떻게 읽는가
- 내 RAG 가 ‘얼마나 좋아졌는지’를 **숫자로** 보여주는 방법

## Step 0 : 설치와 준비
Day 1과 동일하게 Colab에서 진행한다고 가정합니다.


```python
# Colab pre-installed langchain 0.3 / ragas 0.1~0.4 를 ragas 0.2.10 호환 조합으로 정리합니다.
# 처음 실행 시 약 3~5분 걸립니다. 진행률 출력을 보면서 기다리세요 (멈춘 게 아닙니다).

# 1) 기존 langchain / ragas 패키지 제거 — 버전 충돌로 인한 pip resolver 백트래킹 방지
!pip uninstall -y ragas ragas-experimental langchain langchain-core langchain-community langchain-openai langchain-text-splitters langchain-chroma

# 2) 0.2 시리즈 패치 버전까지 핀 설치 — resolver 부담 최소화 (-q 제거해서 진행률 보이게)
!pip install --no-cache-dir \
    "ragas==0.2.10" \
    "langchain==0.2.17" \
    "langchain-core==0.2.43" \
    "langchain-community==0.2.19" \
    "langchain-openai==0.1.25" \
    "langchain-text-splitters==0.2.4" \
    "langchain-chroma==0.1.4" \
    pypdf chromadb tiktoken sentence-transformers datasets nest_asyncio pandas
```


```python
import os
# chromadb 익명 통계 전송 끄기 — posthog SDK 인자 충돌로 ERROR 로그가 뜨는 것 방지
os.environ["ANONYMIZED_TELEMETRY"] = "False"

import nest_asyncio
nest_asyncio.apply()  # RAGAS가 Colab의 비동기 이벤트 루프와 충돌하지 않도록
```


```python
from google.colab import userdata
os.environ['OPENAI_API_KEY'] = userdata.get('OPENAI_KEY')
```

## Step 1 : KorQuAD v1 위에서 Naive RAG 베이스라인 만들기

Day 1에서 만든 RAG 파이프라인을 한국어 QA 벤치마크 **KorQuAD v1** 위에 다시 한 번 올립니다. 이후 단계는 모두 이 베이스라인 위에 ‘덧붙이는’ 방식입니다.

- HuggingFace `datasets` 로 KorQuAD v1 자동 다운로드 (별도 PDF 업로드 불필요)
- 일부만 샘플링해 토큰 비용 통제 (unique context 약 200개)
- Embedding → VectorStore → Retriever → LLM
- 검색 전략은 단순 `similarity` (top-k)

**📥 데이터셋**: <https://huggingface.co/datasets/KorQuAD/squad_kor_v1>


```python
from datasets import load_dataset
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
import tiktoken, random

tokenizer = tiktoken.get_encoding("cl100k_base")
def tiktoken_len(text):
    return len(tokenizer.encode(text))

# 1) 데이터셋 로드 + 2000개 샘플링 + context 중복 제거 → unique 약 800개
#    (Vector DB 가 크면 Reranker 의 정밀도 개선 효과가 더 또렷하게 보입니다.
#     인덱싱 토큰 비용 약 0.01 USD 추가)
raw_ds = load_dataset("squad_kor_v1", split="validation").shuffle(seed=42).select(range(2000))

unique = {}
for ex in raw_ds:
    if ex["context"] not in unique:
        unique[ex["context"]] = ex["title"]
context_docs = [Document(page_content=c, metadata={"title": t}) for c, t in unique.items()]

# 2) chunk 단위로 분할 (KorQuAD context는 짧지만 길이 균질화를 위해 splitter 사용)
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0, length_function=tiktoken_len)
docs = splitter.split_documents(context_docs)

# 3) Embedding & Chroma 적재 — chunk 약 800개를 한 번에 넣으면 chromadb 의 batch limit
#    (Colab 환경에서 보통 5461) 또는 OpenAI rate limit 에 걸릴 수 있어
#    100개씩 배치로 add_documents 합니다.
embedding = OpenAIEmbeddings(model="text-embedding-3-small")
db = Chroma(embedding_function=embedding)
BATCH = 100
for i in range(0, len(docs), BATCH):
    db.add_documents(docs[i:i+BATCH])

# 4) Retriever (Naive: similarity)
naive_retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# 5) LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
print(f"베이스라인 준비 완료 — unique context: {len(context_docs)}, chunks: {len(docs)}")
```

베이스라인 RAG로 간단한 질의를 던져 답이 나오는지 확인합니다.


```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

RAG_PROMPT = ChatPromptTemplate.from_template(
    "다음 문서를 참고해 질문에 한국어로 간결하게 답하세요. 문서에 없는 내용은 만들지 마세요.\n\n"
    "[문서]\n{context}\n\n"
    "[질문]\n{question}\n\n"
    "[답변]"
)

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

naive_chain = (
    {"context": naive_retriever | format_docs,
     "question": RunnablePassthrough()}
    | RAG_PROMPT
    | llm
    | StrOutputParser()
)

# 데이터셋에서 첫 질문 하나를 뽑아 테스트
TEST_Q = raw_ds[0]["question"]
print("Q:", TEST_Q)
print("A:", naive_chain.invoke(TEST_Q))
```

## Step 2 : Pre-retrieval 강화 — Multi-Query Retrieval

사용자가 던진 질문 하나로만 검색하면 ‘다른 표현’으로 적힌 정답을 놓칠 수 있습니다. **Multi-Query Retrieval**은 LLM에게 ‘같은 의도의 다른 질문 N개’를 만들게 시킨 뒤, 각 질문으로 병렬 검색하고 결과를 합칩니다.

LangChain은 이를 한 클래스로 제공합니다.


```python
from langchain.retrievers.multi_query import MultiQueryRetriever
import logging
logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=db.as_retriever(search_kwargs={"k": 3}),
    llm=ChatOpenAI(model="gpt-4o-mini", temperature=0),
)

# 어떤 ‘유사 질문’으로 확장되는지 로그로 확인 가능
docs_mq = multi_query_retriever.invoke(TEST_Q)
print(f"검색된 문서 수: {len(docs_mq)}")
print("---")
print(docs_mq[0].page_content[:300])
```

## Step 2.5 : RAG-Fusion — Multi-Query + RRF로 묶어내기

Day2_1 노트에서 “꼭 짚고 가라”고 했던 패턴 중 하나가 **RAG-Fusion** 입니다. Step 2의 Multi-Query는 ‘유사 질문 N개로 병렬 검색’ 까지만 했는데, **RAG-Fusion** 은 그 N개 검색 결과를 **Reciprocal Rank Fusion (RRF)** 라는 간단한 공식으로 합쳐 ‘여러 쿼리에서 공통으로 상위에 떴던 문서’ 를 최상단으로 끌어올립니다.

RRF 점수 공식:

$$
\text{score}(d) = \sum_{i=1}^{N} \frac{1}{k + \text{rank}_i(d)}
$$

- $\text{rank}_i(d)$ : i번째 쿼리의 결과에서 문서 $d$ 의 순위 (1부터)
- $k$ : 스무딩 상수 (관례적으로 60)

아래 셀에서는 (1) sub-query 생성, (2) 각 sub-query 로 검색, (3) **RRF 함수는 여러분이 직접 채우기**, (4) 결과 확인까지 한 번에 해봅니다.


```python
from collections import defaultdict

# (1) sub-query 생성 — Multi-Query 가 내부적으로 하는 일을 명시적으로 노출 (한국어)
SUBQUERY_PROMPT = ChatPromptTemplate.from_template(
    "당신은 검색 보조 AI 입니다. 다음 질문과 의미는 같지만 표현이 다른 4개의 한국어 검색 쿼리를 만드세요. "
    "오직 4개의 쿼리만 한 줄에 하나씩 출력하고, 번호나 다른 설명은 붙이지 마세요.\n\n질문: {question}"
)

def fan_out_queries(question, n=4):
    raw = (SUBQUERY_PROMPT | llm | StrOutputParser()).invoke({"question": question})
    return [q.strip() for q in raw.split("\n") if q.strip()][:n]


# (2) RRF 함수
def reciprocal_rank_fusion(results_per_query, k=60, top_k=3):
    """
    results_per_query : List[List[Document]]  쿼리별 검색 결과(순위 순).
    k                 : RRF smoothing 상수 (관례적으로 60).
    top_k             : 최종 반환할 문서 개수.

    핵심 아이디어:
    - 여러 검색 결과에서 반복해서 상위에 등장한 문서일수록 점수가 커진다.
    - rank는 enumerate 때문에 0부터 시작하므로 실제 순위처럼 쓰기 위해 +1 한다.
    """
    scores = defaultdict(float)
    docs_by_key = {}

    # TODO 1 해결: 쿼리별 검색 결과를 순회하며 문서마다 RRF 점수를 누적
    for docs in results_per_query:
        for rank, doc in enumerate(docs):
            key = doc.page_content
            scores[key] += 1.0 / (k + rank + 1)
            docs_by_key[key] = doc

    # TODO 2 해결: 점수 높은 순으로 정렬한 뒤 상위 top_k개 문서 반환
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [docs_by_key[key] for key, _ in ranked[:top_k]]


# (3) 한 번 돌려보기
sub_queries = fan_out_queries(TEST_Q)
print(f"확장 질문 {len(sub_queries)}개:")
for q in sub_queries:
    print(" -", q)

results_per_q = [db.similarity_search(q, k=5) for q in sub_queries]
fused = reciprocal_rank_fusion(results_per_q, k=60, top_k=3)

print("\nRAG-Fusion top-1 문서:")
print(fused[0].page_content[:300] if fused else "(결과가 없습니다)")
```

## Step 3 : 패턴 ② HyDE — 가상의 ‘정답’으로 진짜 정답 찾기

질문은 짧은 의문문, 정답은 긴 평서문이라 둘의 임베딩이 의외로 멀 수 있습니다. **HyDE(Hypothetical Document Embeddings)** 는 검색 전에 LLM에게 ‘가상의 정답’을 쓰게 한 뒤, 그 가상 답변을 임베딩해서 검색합니다.

직접 구현해 보겠습니다.


```python
HYDE_PROMPT = ChatPromptTemplate.from_template(
    "당신은 해당 분야 전문가입니다. 다음 질문에 대해 그럴듯한 한국어 답변 한 문단을 작성하세요. "
    "확실하지 않다면 가장 합리적인 추측을 적어주세요.\n\n"
    "질문: {question}\n\n가상 답변:"
)

hyde_generator = HYDE_PROMPT | llm | StrOutputParser()

def hyde_retrieve(question, k=3):
    """질문 → 가상의 답변 → 가상 답변을 임베딩해 검색"""
    hypothetical = hyde_generator.invoke({"question": question})
    return db.similarity_search(hypothetical, k=k), hypothetical

docs_hyde, hyp = hyde_retrieve(TEST_Q)
print("가상 답변(HyDE):\n", hyp[:300], "\n---")
print("검색된 문서 수:", len(docs_hyde))
print("첫 문서:", docs_hyde[0].page_content[:200])
```

## Step 4 : Post-retrieval 강화 — Cross-Encoder Reranking (multilingual)

검색 결과를 그대로 LLM 에 넘기지 않고, **Cross-encoder reranker** 가 (질문, 문단)을 함께 보면서 진짜 관련도를 다시 점수화합니다. 정밀도가 15~30% 개선되는 게 일반적인 보고입니다.

한국어 문서를 다루고 있으므로 다국어를 지원하는 cross-encoder 를 사용합니다. `BAAI/bge-reranker-v2-m3` 는 한국어를 포함한 100개 이상 언어에서 동작합니다. 처음 실행 시 모델 다운로드(~2GB)가 발생합니다.


```python
from sentence_transformers import CrossEncoder

# 다국어 cross-encoder (한국어 포함)
reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")

def rerank(query, docs, top_k=3):
    """검색된 docs 를 cross-encoder 로 다시 점수화해 상위 top_k 만 반환"""
    pairs = [(query, d.page_content) for d in docs]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [d for d, _ in ranked[:top_k]]

candidates = db.as_retriever(search_kwargs={"k": 10}).invoke(TEST_Q)
top3 = rerank(TEST_Q, candidates, top_k=3)
print(f"후보 {len(candidates)}개 → Reranker 로 상위 3개 선별")
print("최상위 문서:", top3[0].page_content[:200])
```

## Step 5 : Advanced RAG 체인 조립

위에서 만든 컴포넌트들을 하나의 체인으로 묶습니다. **‘넓게 검색 → Reranker로 좁히기 → LLM 답변’** 패턴이 가장 흔히 쓰입니다.


```python
# 빠른 실습 모드에서는 reranker 후보 수를 줄여 Step 6 실행 시간을 줄인다.
# FULL 비교가 필요하면 RERANK_CANDIDATE_K = 10 으로 변경한다.
RERANK_CANDIDATE_K = 5

def advanced_rag(question):
    # 1) 후보를 넓게 검색한다. 원래는 k=10을 많이 쓰지만, 실습 속도를 위해 기본 5개로 둔다.
    candidates = db.as_retriever(search_kwargs={"k": RERANK_CANDIDATE_K}).invoke(question)
    # 2) Cross-encoder 로 진짜 관련도 재정렬 후 상위 3개
    top = rerank(question, candidates, top_k=min(3, len(candidates)))
    # 3) 프롬프트에 컨텍스트로 주입 → 답변
    context = format_docs(top)
    answer = (RAG_PROMPT | llm | StrOutputParser()).invoke(
        {"context": context, "question": question})
    return answer, top

ans_adv, ctx_adv = advanced_rag(TEST_Q)
print("Advanced RAG 답변:\n", ans_adv)
```

## Step 5.5 : Self-RAG — 검색 필요성 판단 + 답변 자가 비평

Day2_1 노트에서 강조한 또 하나의 핵심 패턴, **Self-RAG** 입니다. Self-RAG의 핵심은 **LLM이 검색·답변 과정에 스스로 비평(critique)을 끼워 넣는다**는 점입니다.

이번 셀에서는 공식 Self-RAG 모델을 따로 받지 않고, **세 개의 작은 LLM 프롬프트**로 같은 흐름을 흉내내 봅니다.

1. **Retrieve 결정** — 질문이 들어오면, 외부 검색이 필요한지 LLM이 먼저 판단합니다. (`YES`/`NO` 한 단어)
2. **답변 생성** — `YES` 면 일반 RAG, `NO` 면 검색 없이 LLM 단독 답변.
3. **답변 자가 비평** — 생성된 답변이 컨텍스트에 충분히 근거하는지 LLM이 점검합니다. (`SUPPORTED` / `NOT_SUPPORTED`)
4. **보완 재시도** — `NOT_SUPPORTED` 면 Step 3의 **HyDE** 로 검색 쿼리를 바꿔 한 번 더 시도합니다.

코드 골격은 제공해 두었고, **두 군데 핵심 프롬프트만 여러분이 직접 채워주세요.**


```python
# Self-RAG : retrieve 판단 + 자가 비평 + HyDE 재시도

# (1) 검색 필요성 판단 프롬프트
RETRIEVE_DECISION_PROMPT = ChatPromptTemplate.from_template(
    "당신은 RAG 시스템의 검색 필요성 판단기입니다.\n"
    "사용자 질문이 주어진 문서, 최신 정보, 특정 데이터셋, 근거 문서 검색이 필요한 질문이면 YES를 출력하세요.\n"
    "일반 상식, 간단한 계산, 용어 정의처럼 외부 문서 없이 답할 수 있으면 NO를 출력하세요.\n"
    "반드시 YES 또는 NO 한 단어만 출력하세요.\n\n"
    "[질문]\n{question}\n\n"
    "[판단]"
)

# (2) 답변 자가 비평 프롬프트
CRITIQUE_PROMPT = ChatPromptTemplate.from_template(
    "당신은 RAG 답변 검증기입니다.\n"
    "아래 [문서]만 근거로 [답변]이 충분히 뒷받침되는지 판단하세요.\n"
    "답변의 핵심 내용이 문서에 근거하면 SUPPORTED를 출력하세요.\n"
    "문서에 없는 내용, 추측, 과장이 포함되어 있으면 NOT_SUPPORTED를 출력하세요.\n"
    "반드시 SUPPORTED 또는 NOT_SUPPORTED 한 단어만 출력하세요.\n\n"
    "[문서]\n{context}\n\n"
    "[답변]\n{answer}\n\n"
    "[판단]"
)


def self_rag(question, max_retries=1, verbose=True):
    decision = (RETRIEVE_DECISION_PROMPT | llm | StrOutputParser()).invoke(
        {"question": question}).strip().upper()
    if verbose:
        print(f"[1] Retrieve 필요? -> {decision}")

    if decision.startswith("NO"):
        ans = llm.invoke(question).content
        if verbose:
            print("[2] LLM 단독 답변 사용")
        return ans, []

    docs = db.as_retriever(search_kwargs={"k": 3}).invoke(question)

    for attempt in range(max_retries + 1):
        answer = (RAG_PROMPT | llm | StrOutputParser()).invoke(
            {"context": format_docs(docs), "question": question})
        critique = (CRITIQUE_PROMPT | llm | StrOutputParser()).invoke(
            {"context": format_docs(docs), "answer": answer}).strip().upper()
        if verbose:
            print(f"[3] 시도 {attempt+1} — 자가 비평: {critique}")

        if "NOT" not in critique:
            return answer, docs

        if attempt < max_retries:
            hyp = hyde_generator.invoke({"question": question})
            docs = db.similarity_search(hyp, k=3)
            if verbose:
                print("[4] NOT_SUPPORTED -> HyDE 가상 답변으로 재검색")

    # 모든 재시도 후에도 NOT_SUPPORTED라면 마지막 답변을 그대로 내보내지 않는다.
    # Self-RAG의 목적은 모르는 것을 안전하게 거절/보류하는 것이기 때문이다.
    return "검색된 문서만으로는 답변을 충분히 확인할 수 없습니다.", docs


ans_sr, ctx_sr = self_rag(TEST_Q)
print("\n=== Self-RAG 최종 답변 ===")
print(ans_sr)
```

## Step 6 : RAGAS 평가용 데이터셋 만들기

RAGAS 는 네 가지 자료가 필요합니다.
- `user_input` — 사용자 질문
- `response`   — RAG 가 생성한 답변
- `retrieved_contexts` — RAG 가 참고한 문서들
- `reference`  — 모범 답안 (Ground Truth)

**KorQuAD 는 사람이 작성한 정답이 데이터셋에 이미 포함**되어 있어, `reference` 를 따로 작성할 필요 없이 그대로 가져다 씁니다. 같은 질문 셋을 **Naive RAG** 와 **Advanced RAG** 두 가지로 풀고 결과를 비교합니다.

빠른 실습을 위해 기본 평가 질문은 **3개**만 사용합니다.
- 빠른 확인: `FAST_MODE = True`, `EVAL_N = 3`
- 최종 비교: `FAST_MODE = False`, `EVAL_N = 20` 이상

이 셀은 질문마다 Naive RAG와 Advanced RAG를 실제로 호출하므로 오래 걸릴 수 있습니다. 진행 상황이 보이도록 `print`를 넣었습니다.


```python
# 평가용 질문/정답 자동 추출 (KorQuAD)
# 빠른 실습 모드: 3개만 먼저 돌려서 전체 파이프라인이 정상인지 확인한다.
# 최종 비교가 필요하면 FAST_MODE = False 로 바꾸고 EVAL_N_FULL 값을 늘린다.

FAST_MODE = True
EVAL_N_FAST = 3
EVAL_N_FULL = 20
EVAL_N = EVAL_N_FAST if FAST_MODE else EVAL_N_FULL

# raw_ds에서 평가 샘플을 가져온다.
eval_samples = list(raw_ds)[:EVAL_N]
questions = [ex["question"] for ex in eval_samples]
ground_truths = [ex["answers"]["text"][0] for ex in eval_samples]

print(f"평가 질문 수: {EVAL_N}개")
print("FAST_MODE:", FAST_MODE)

# Naive RAG / Advanced RAG 답변과 컨텍스트를 한 번에 수집한다.
# 진행률을 출력해서 멈춘 것처럼 보이지 않게 한다.
naive_answers, naive_contexts = [], []
adv_answers, adv_contexts = [], []

for i, q in enumerate(questions, 1):
    print(f"\n[{i}/{EVAL_N}] 질문 처리 중")
    print("질문:", q[:80])

    # 1) Naive RAG
    print(" - Naive RAG 실행")
    ctx = naive_retriever.invoke(q)
    a = (RAG_PROMPT | llm | StrOutputParser()).invoke(
        {"context": format_docs(ctx), "question": q}
    )
    naive_answers.append(a)
    naive_contexts.append([d.page_content for d in ctx])

    # 2) Advanced RAG
    print(f" - Advanced RAG 실행 (rerank 후보 k={RERANK_CANDIDATE_K})")
    a_adv, ctx_adv = advanced_rag(q)
    adv_answers.append(a_adv)
    adv_contexts.append([d.page_content for d in ctx_adv])

print(f"\n데이터셋 준비 완료 — {EVAL_N}개 질문 × 2개 파이프라인")

```


```python
from datasets import Dataset

def make_dataset(answers, contexts):
    return Dataset.from_dict({
        "user_input":         questions,
        "response":           answers,
        "retrieved_contexts": contexts,
        "reference":          ground_truths,
    })

naive_ds = make_dataset(naive_answers, naive_contexts)
adv_ds   = make_dataset(adv_answers,   adv_contexts)
```

## Step 7 : RAGAS로 4대 지표 계산하기

Judge LLM은 `gpt-4o-mini`로, 임베딩은 `text-embedding-3-small`로 설정합니다.
(Judge에 더 강한 모델을 쓰면 채점은 더 정교해지지만 비용이 늘어납니다.)


```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness, answer_relevancy,
    context_precision, context_recall,
)

judge_llm  = ChatOpenAI(model="gpt-4o-mini", temperature=0)
judge_emb  = OpenAIEmbeddings(model="text-embedding-3-small")
metrics    = [faithfulness, answer_relevancy,
              context_precision, context_recall]

print(f"RAGAS 평가 시작 — 샘플 {EVAL_N}개, 지표 {len(metrics)}개")
print("처음에는 3개 샘플로 정상 실행되는지 확인하세요. 전체 평가는 시간이 더 오래 걸립니다.\n")

print("=== Naive RAG 채점 ===")
naive_result = evaluate(
    naive_ds,
    metrics=metrics,
    llm=judge_llm,
    embeddings=judge_emb,
    raise_exceptions=False,
)

print("=== Advanced RAG 채점 ===")
adv_result = evaluate(
    adv_ds,
    metrics=metrics,
    llm=judge_llm,
    embeddings=judge_emb,
    raise_exceptions=False,
)

```


```python
import pandas as pd
pd.set_option('display.max_colwidth', None)

naive_df = naive_result.to_pandas()
adv_df   = adv_result.to_pandas()

def summary(df, label):
    cols = ["faithfulness", "answer_relevancy",
            "context_precision", "context_recall"]
    avg = df[cols].mean()
    avg.name = label
    return avg

compare = pd.concat([summary(naive_df, "Naive RAG"),
                     summary(adv_df,   "Advanced RAG")], axis=1)
print(compare.round(3))
print("\nDelta (Advanced - Naive):")
print((compare["Advanced RAG"] - compare["Naive RAG"]).round(3))
```

### 결과 해석 가이드

위 비교표를 처음 보면 **‘Advanced 가 더 나쁜 거 아닌가?’** 라는 착각을 하기 쉽습니다. KorQuAD 위에서의 결과 해석 방법을 정리합니다.

**1. `context_precision` 의 개선 (+) 이 Advanced RAG 의 핵심 효과**
검색 결과의 ‘상단’에 정답 문단을 두는 일을 Reranker 가 잘 했다는 의미. Δ가 0.05~0.15 정도면 잘 작동.

**2. `context_recall = 1.0` 으로 포화될 수 있다**
unique context 가 800개 정도면 Naive top-3 에도 정답이 거의 항상 들어옵니다. 이 지표는 더 큰 DB(수만 문서)에서 차이가 드러납니다.

**3. `faithfulness` 가 살짝 떨어질 수 있다**
Reranker 가 컨텍스트를 ‘짧고 집중’ 시키면 LLM이 그 좁은 정보에서 답을 만들 때 일부 주장이 “미뒷받침” 으로 채점되어 점수가 약간 내려갈 수 있음. **정상 범위 (-0.1 이내)**.

**4. `answer_relevancy` 가 0.2~0.4 로 낮은 이유 — KorQuAD 의 구조적 특성**
KorQuAD 정답은 *‘대중교통체계’* 같이 한 단어~한 구절. RAG 답변도 짧게 나오는데, RAGAS 의 `answer_relevancy` 는 **답변에서 질문을 역추론**해 원래 질문과의 유사도를 계산합니다. 답변이 한 단어면 역추론이 흐려져 점수가 낮아집니다. **모델 잘못이 아닌 데이터셋 특성**.

**5. 표본 20개로도 Δ가 ±0.05 이내면 ‘차이 없음’으로 봐야 한다**
20문항에서 ±0.05 는 표본 noise. 더 확실한 판단이 필요하면 `scipy.stats.ttest_rel` 로 통계 검정을 하거나 50~100문항으로 늘려야 합니다.

**6. 한국어 짧은 정답 벤치마크의 한계**
KorQuAD/KLUE-MRC 처럼 정답이 짧은 extractive QA 벤치마크는 `context_precision` 위주로 평가 효과를 봐야 하고, `answer_relevancy` 는 절대값보다 **Naive 대비 상대 변화**로 읽어야 합니다.

### Quiz
위 표에서 Advanced RAG가 가장 크게 개선한 지표는 무엇인가요? 그리고 그 지표는 우리가 적용한 **어떤 기법**과 가장 직접적으로 연결될까요?

**Answer (예시)**:
보통 `context_precision`이 가장 크게 오릅니다. 이는 우리가 추가한 **Reranker**가 ‘진짜 관련도가 높은 문서를 상위에 두는 일’을 잘 했다는 의미입니다.  `context_recall`은 **Multi-Query**가 검색 폭을 넓혔다면 같이 오릅니다.  `faithfulness`와 `answer_relevancy`는 컨텍스트 품질이 올라가면 부수적으로 개선됩니다.

## Step 8 : (선택) 평가 데이터를 LLM으로 자동 생성하기

현업에서는 모범 답안(`reference`)을 사람이 직접 작성하는 게 가장 큰 부담입니다.
RAGAS는 **원본 문서만 주면 Question·Reference·Context 한 세트를 자동으로 만들어 주는** 기능을 제공합니다.
자세한 사용법은 공식 문서를 참고하세요.

https://docs.ragas.io/en/stable/getstarted/rag_testset_generation/

---
# 추가 실습 — KLUE-MRC 한국어 뉴스 MRC 벤치마크로 RAG 평가하기

메인 실습은 위키 기반 **KorQuAD v1** 으로 진행했습니다. 이번 추가 실습은 도메인을 바꿔, **한국어 뉴스 기사 기반의 MRC 벤치마크 KLUE-MRC** 위에서 같은 파이프라인을 처음부터 다시 조립해 봅니다.

**KLUE-MRC**
- 카카오·네이버 등 한국 NLP 팀이 함께 만든 한국어 표준 벤치마크 KLUE 의 MRC 태스크
- 한국어 **뉴스 기사** 기반 (KorQuAD 의 위키와 도메인이 다름)
- 사람이 직접 작성한 정답 포함
- **`is_impossible=True`** 인 답할 수 없는 질문도 일부 포함 → 데이터 필터링이 필요한 도전적 케이스

위키 기반 KorQuAD 와 비교했을 때 어떤 차이(질문 스타일, 검색 난이도, 점수 분포)가 나는지 직접 관찰해 보세요.

이번에도 일부만 샘플링해서 토큰 비용을 통제합니다.
- Vector DB 에 들어갈 unique context: 약 200개
- 평가 질문: 20개
- 예상 비용: GPT-4o-mini 기준 RAGAS 평가까지 합쳐서 약 \$0.10 ~ \$0.20

**📥 데이터셋 다운로드 / 출처**
- HuggingFace `datasets` 자동 다운로드: <https://huggingface.co/datasets/klue>
- KLUE 공식 사이트: <https://klue-benchmark.com/>
- KLUE 논문: <https://arxiv.org/abs/2105.09680>

> 다른 데이터셋으로 한 번 더 해보고 싶다면:
> - MIRACL 한국어: <https://huggingface.co/datasets/miracl/miracl> (config: `ko`)
> - 영어 SQuAD: <https://huggingface.co/datasets/rajpurkar/squad>

### Step A. 데이터셋 로드

`datasets` 라이브러리로 KLUE-MRC 를 한 줄에 받아옵니다. KLUE 는 여러 sub-task 가 있는 멀티태스크 벤치마크라서 config 이름 `"mrc"` 를 명시해야 합니다.

데이터셋 페이지: <https://huggingface.co/datasets/klue>


```python
from datasets import load_dataset

ds_klue = load_dataset("klue", "mrc", split="validation")
print(ds_klue)
print("\n--- 샘플 1건 ---")
print({k: ds_klue[0][k] for k in ds_klue.column_names})
```

### Step B. Context 추출 + 중복 제거 (+ is_impossible 필터링)

KLUE-MRC 에는 KorQuAD 에는 없는 **`is_impossible=True`** 케이스가 섞여 있습니다 (= context 만 보고는 답할 수 없는 질문). 평가용 ground_truth 가 비어 있으면 RAGAS 의 `context_recall` 이 깨지므로, 답이 있는 샘플만 남기세요.

- `ds_klue.filter(lambda x: not x["is_impossible"])` 로 답 있는 것만 추리고
- `shuffle(seed=42).select(range(300))` 으로 300개 샘플링
- 그 중 `context` 필드 기준으로 중복 제거 (보통 150~200개)
- 각각을 `Document(page_content=..., metadata={"title": ex["title"]})` 로 감싸 `context_docs` 에 담기


```python
# 안내에 따라 context_docs 리스트 만들기
# 1) 답할 수 없는 질문(is_impossible=True)을 제거
# 2) 300개만 샘플링
# 3) context 기준으로 중복 제거
# 4) LangChain Document 객체로 변환

import random
from langchain_core.documents import Document

klue_answerable = ds_klue.filter(lambda x: not x["is_impossible"])
klue_samples = klue_answerable.shuffle(seed=42).select(
    range(min(300, len(klue_answerable)))
)

unique_contexts_klue = {}
for ex in klue_samples:
    context = ex["context"]
    if context not in unique_contexts_klue:
        unique_contexts_klue[context] = {
            "title": ex.get("title", ""),
            "source": ex.get("source", ""),
        }

context_docs = [
    Document(page_content=context, metadata=metadata)
    for context, metadata in unique_contexts_klue.items()
]

print("답 있는 KLUE 샘플 수:", len(klue_samples))
print("중복 제거 후 context 문서 수:", len(context_docs))
print("\n첫 문서 미리보기:")
print(context_docs[0].page_content[:200])
```

### Step C. Embedding + VectorStore

메인 실습에서 만든 `embedding` (`OpenAIEmbeddings(model="text-embedding-3-small")`) 을 그대로 재사용해, `context_docs` 로 새 Chroma DB `db_klue` 를 만드세요. (메인 실습의 `db` 변수를 덮어쓰지 마세요. 비교가 안 됩니다.)

> ⚠️ **batch 적재 필수** — KLUE-MRC 의 뉴스 context 는 평균 토큰 수가 커서, 150개 이상을 한 번에 `Chroma.from_documents` 로 넘기면 OpenAI embeddings 의 **300k 토큰/요청 한도** 에 걸려 `BadRequestError` 가 납니다. 메인 cell 8 처럼 100개씩 batch 로 `add_documents` 호출하세요:
> ```python
> db_klue = Chroma(embedding_function=embedding)
> BATCH = 100
> for i in range(0, len(context_docs), BATCH):
>     db_klue.add_documents(context_docs[i:i+BATCH])
> ```

> 인덱싱 토큰 비용: 약 200개 context × 평균 600 토큰 ≈ **120k 토큰** (≈ \$0.003)


```python
# db_klue 만들기
# 한 번에 넣지 않고 100개씩 나누어 넣어 OpenAI Embedding 요청 토큰 한도를 피한다.

db_klue = Chroma(embedding_function=embedding)

BATCH = 100
for i in range(0, len(context_docs), BATCH):
    db_klue.add_documents(context_docs[i:i+BATCH])
    print(f"적재 완료: {min(i + BATCH, len(context_docs))}/{len(context_docs)}")

print("KLUE Chroma DB 준비 완료")
```

### Step D. 평가용 질문/정답 세트 추출

Step B 에서 필터링·샘플링한 데이터 중 **앞에서 20개**를 평가용으로 떼어내세요.

- `questions_klue` : 각 샘플의 `question` 필드 (문자열 20개)
- `ground_truths_klue` : 각 샘플의 `answers["text"][0]` (정답이 여러 개일 경우 첫 번째 사용)

> 참고: KLUE-MRC 는 정답이 한 구절~한 문장 단위의 **extractive QA** 입니다. 짧은 정답은 RAGAS 의 `context_recall` 변동성을 키우는 경향이 있으니, 평균을 함께 봐주세요.


```python
# 평가용 질문/정답 세트 만들기
# 빠른 실습에서는 3개만 먼저 평가한다. 최종 비교가 필요하면 EVAL_N_KLUE_FULL 값을 늘린다.

FAST_MODE_KLUE = True
EVAL_N_KLUE_FAST = 3
EVAL_N_KLUE_FULL = 20
EVAL_N_KLUE = EVAL_N_KLUE_FAST if FAST_MODE_KLUE else EVAL_N_KLUE_FULL

eval_samples_klue = list(klue_samples)[:EVAL_N_KLUE]

questions_klue = [ex["question"] for ex in eval_samples_klue]
ground_truths_klue = [ex["answers"]["text"][0] for ex in eval_samples_klue]

print("FAST_MODE_KLUE:", FAST_MODE_KLUE)
print("questions_klue 길이:", len(questions_klue))
print("ground_truths_klue 길이:", len(ground_truths_klue))
print("\n예시 질문:", questions_klue[0])
print("예시 정답:", ground_truths_klue[0])

```

### Step E. Naive RAG 베이스라인 (KLUE)

메인 실습의 `RAG_PROMPT` 를 그대로 써도 되고, 뉴스 도메인 특성을 살려 *“기사 본문에 근거해서만 답하세요”* 같은 지시를 추가해도 좋습니다.

- `naive_retriever_klue = db_klue.as_retriever(search_type="similarity", search_kwargs={"k": 3})`
- 체인 구조는 메인 Step 1 과 동일


```python
# KLUE용 Naive RAG 베이스라인 만들기
# 기존 RAG_PROMPT를 그대로 써도 되지만, 뉴스 기사 기반이라는 점을 더 명확히 적었다.

RAG_PROMPT_KLUE = ChatPromptTemplate.from_template(
    "다음 뉴스 기사 문서를 참고해 질문에 한국어로 간결하게 답하세요. "
    "문서에 없는 내용은 만들지 마세요.\n\n"
    "[문서]\n{context}\n\n"
    "[질문]\n{question}\n\n"
    "[답변]"
)

naive_retriever_klue = db_klue.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

naive_chain_klue = (
    {"context": naive_retriever_klue | format_docs,
     "question": RunnablePassthrough()}
    | RAG_PROMPT_KLUE
    | llm
    | StrOutputParser()
)

print("Q:", questions_klue[0])
print("A:", naive_chain_klue.invoke(questions_klue[0]))
```

### Step F. Multi-Query Retrieval

메인 Step 2 와 동일하게 `MultiQueryRetriever.from_llm(...)` 으로 KLUE 검색기를 감싸세요. 한국어 질문이 들어가면 gpt-4o-mini 가 한국어로 유사 질문을 만들어 줍니다.

확장 질문 로깅을 켜서 어떤 한국어 변형 질문이 만들어지는지 직접 눈으로 확인하세요.


```python
# KLUE용 Multi-Query Retriever
# 질문 하나를 여러 표현으로 바꿔 검색 폭을 넓힌다.

import logging
from langchain.retrievers.multi_query import MultiQueryRetriever

logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

multi_query_retriever_klue = MultiQueryRetriever.from_llm(
    retriever=db_klue.as_retriever(search_kwargs={"k": 3}),
    llm=ChatOpenAI(model="gpt-4o-mini", temperature=0),
)

docs_mq_klue = multi_query_retriever_klue.invoke(questions_klue[0])

print("질문:", questions_klue[0])
print(f"검색된 문서 수: {len(docs_mq_klue)}")
print("--- 첫 문서 미리보기 ---")
print(docs_mq_klue[0].page_content[:300])
```

### Step G. HyDE 직접 구현

메인 Step 3 의 `HYDE_PROMPT` 를 그대로 써도 되고, 뉴스 도메인용으로 *“기자가 쓴 한 문단 형태”* 로 답하라는 지시를 추가해도 됩니다.

`hyde_retrieve_klue(question, k=3)` 함수를 만들고 `db_klue` 위에서 동작하도록 하세요.


```python
# KLUE용 HyDE
# 질문에 바로 임베딩을 거는 대신, LLM이 만든 '가상 뉴스 답변'을 임베딩해서 검색한다.

HYDE_PROMPT_KLUE = ChatPromptTemplate.from_template(
    "당신은 한국어 뉴스 기사를 읽고 답하는 전문가입니다. "
    "다음 질문에 대한 그럴듯한 답변을 뉴스 기사 문장처럼 한 문단으로 작성하세요. "
    "확실하지 않다면 가장 합리적인 추측을 적되, 과도하게 길게 쓰지 마세요.\n\n"
    "질문: {question}\n\n"
    "가상 답변:"
)

hyde_generator_klue = HYDE_PROMPT_KLUE | llm | StrOutputParser()

def hyde_retrieve_klue(question, k=3):
    hypothetical = hyde_generator_klue.invoke({"question": question})
    docs = db_klue.similarity_search(hypothetical, k=k)
    return docs, hypothetical

docs_hyde_klue, hyp_klue = hyde_retrieve_klue(questions_klue[0], k=3)

print("질문:", questions_klue[0])
print("\n가상 답변(HyDE):")
print(hyp_klue[:300])
print("\n검색된 문서 수:", len(docs_hyde_klue))
print("--- 첫 문서 미리보기 ---")
print(docs_hyde_klue[0].page_content[:300])
```

### Step H. Multilingual Cross-encoder Reranker

메인 Step 4 에서 이미 `BAAI/bge-reranker-v2-m3` 같은 다국어 reranker 를 사용하고 있습니다. 추가 실습에서는:

- 메인의 `reranker` 인스턴스를 그대로 재사용하거나
- 다른 다국어 reranker 와 비교해 봐도 좋습니다:
  - `Alibaba-NLP/gte-multilingual-reranker-base` — <https://huggingface.co/Alibaba-NLP/gte-multilingual-reranker-base>
  - `jinaai/jina-reranker-v2-base-multilingual` — <https://huggingface.co/jinaai/jina-reranker-v2-base-multilingual>

`rerank_klue(query, docs, top_k=3)` 함수를 만드세요. (메인 Step 4 의 `rerank` 와 동일 구조)


```python
# KLUE용 Cross-Encoder Reranker
# 기존 reranker가 이미 만들어져 있으면 재사용하고, 없으면 새로 로드한다.

from sentence_transformers import CrossEncoder

try:
    reranker_klue = reranker
    print("기존 reranker를 재사용합니다.")
except NameError:
    reranker_klue = CrossEncoder("BAAI/bge-reranker-v2-m3")
    print("KLUE용 reranker를 새로 로드했습니다.")

def rerank_klue(query, docs, top_k=3):
    pairs = [(query, d.page_content) for d in docs]
    scores = reranker_klue.predict(pairs)
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [d for d, _ in ranked[:top_k]]

candidates_klue = db_klue.as_retriever(search_kwargs={"k": 10}).invoke(questions_klue[0])
top3_klue = rerank_klue(questions_klue[0], candidates_klue, top_k=3)

print(f"후보 {len(candidates_klue)}개 → Reranker로 상위 {len(top3_klue)}개 선별")
print("--- 최상위 문서 미리보기 ---")
print(top3_klue[0].page_content[:300])
```

### Step I. Advanced RAG 체인 (넓게 → Rerank → LLM)

메인 Step 5 흐름과 동일.
1. `db_klue.as_retriever(search_kwargs={"k": 10}).invoke(question)` 로 후보 10개
2. `rerank_klue(question, candidates, top_k=3)` 로 좁힘
3. `RAG_PROMPT` + `llm` 으로 답변 생성

함수가 `(answer, top_docs)` 둘 다 반환하도록 만들어 두면 다음 평가 단계에서 그대로 씁니다.


```python
# KLUE용 Advanced RAG
# 넓게 검색 → reranker로 상위 3개만 선택 → LLM 답변 생성
# 빠른 실습 모드에서는 후보 수를 5개로 줄인다. 최종 비교에서는 10으로 늘려도 된다.

RERANK_CANDIDATE_K_KLUE = 5

def advanced_rag_klue(question):
    candidates = db_klue.as_retriever(search_kwargs={"k": RERANK_CANDIDATE_K_KLUE}).invoke(question)
    top_docs = rerank_klue(question, candidates, top_k=min(3, len(candidates)))
    answer = (RAG_PROMPT_KLUE | llm | StrOutputParser()).invoke(
        {"context": format_docs(top_docs), "question": question}
    )
    return answer, top_docs

ans_adv_klue, ctx_adv_klue = advanced_rag_klue(questions_klue[0])

print("Q:", questions_klue[0])
print("\nAdvanced RAG 답변:")
print(ans_adv_klue)
print("\n사용한 context 수:", len(ctx_adv_klue))
print("--- 첫 context 미리보기 ---")
print(ctx_adv_klue[0].page_content[:300])

```

### Step J. RAGAS 로 Naive vs Advanced 비교

메인 Step 6/7 흐름을 KLUE-MRC 변수(`_klue`) 로 옮겨 동일하게 수행하세요.

1. 기본 3개 질문을 Naive / Advanced 파이프라인에 돌려 답변과 컨텍스트 수집
2. `Dataset.from_dict({...})` 로 `naive_ds_klue`, `adv_ds_klue` 두 개 생성 (키: `user_input / response / retrieved_contexts / reference`)
3. `evaluate(..., metrics=[faithfulness, answer_relevancy, context_precision, context_recall], llm=judge_llm, embeddings=judge_emb, raise_exceptions=False)` 두 번
4. 평균표로 비교

메인의 KorQuAD 결과와 점수가 어떻게 다른지 옆에 같이 적어두면 학습 효과가 큽니다.


```python
# KLUE-MRC에서 Naive RAG vs Advanced RAG를 RAGAS로 비교

from datasets import Dataset

# 1) 답변과 컨텍스트 수집
naive_answers_klue, naive_contexts_klue = [], []
adv_answers_klue, adv_contexts_klue = [], []

for i, q in enumerate(questions_klue, 1):
    print(f"\n[{i}/{len(questions_klue)}] KLUE 질문 처리 중")
    print("질문:", q[:80])

    # Naive
    print(" - KLUE Naive RAG 실행")
    ctx = naive_retriever_klue.invoke(q)
    ans = (RAG_PROMPT_KLUE | llm | StrOutputParser()).invoke(
        {"context": format_docs(ctx), "question": q}
    )
    naive_answers_klue.append(ans)
    naive_contexts_klue.append([d.page_content for d in ctx])

    # Advanced
    print(f" - KLUE Advanced RAG 실행 (rerank 후보 k={RERANK_CANDIDATE_K_KLUE})")
    ans_adv, ctx_adv = advanced_rag_klue(q)
    adv_answers_klue.append(ans_adv)
    adv_contexts_klue.append([d.page_content for d in ctx_adv])

print(f"\n답변 생성 완료 — KLUE {len(questions_klue)}개 질문 × 2개 파이프라인")

# 2) RAGAS 입력 Dataset 생성
def make_dataset_klue(answers, contexts):
    return Dataset.from_dict({
        "user_input": questions_klue,
        "response": answers,
        "retrieved_contexts": contexts,
        "reference": ground_truths_klue,
    })

naive_ds_klue = make_dataset_klue(naive_answers_klue, naive_contexts_klue)
adv_ds_klue = make_dataset_klue(adv_answers_klue, adv_contexts_klue)

# 3) RAGAS 평가
print(f"\nRAGAS 평가 시작 — KLUE 샘플 {len(questions_klue)}개, 지표 {len(metrics)}개")

print("=== KLUE Naive RAG 채점 ===")
naive_result_klue = evaluate(
    naive_ds_klue,
    metrics=metrics,
    llm=judge_llm,
    embeddings=judge_emb,
    raise_exceptions=False,
)

print("=== KLUE Advanced RAG 채점 ===")
adv_result_klue = evaluate(
    adv_ds_klue,
    metrics=metrics,
    llm=judge_llm,
    embeddings=judge_emb,
    raise_exceptions=False,
)

# 4) 평균 비교표 출력
naive_df_klue = naive_result_klue.to_pandas()
adv_df_klue = adv_result_klue.to_pandas()

def summary_klue(df, label):
    cols = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    avg = df[cols].mean()
    avg.name = label
    return avg

compare_klue = pd.concat([
    summary_klue(naive_df_klue, "KLUE Naive RAG"),
    summary_klue(adv_df_klue, "KLUE Advanced RAG"),
], axis=1)

print("\n=== KLUE 평균 비교 ===")
print(compare_klue.round(3))

print("\nDelta (Advanced - Naive):")
print((compare_klue["KLUE Advanced RAG"] - compare_klue["KLUE Naive RAG"]).round(3))

```

### Step K. (선택) 좀 더 큰 샘플로 통계적 신뢰도 확보

질문 20개로는 표본 분산이 커서 Naive vs Advanced 차이가 우연일 수도 있습니다. 토큰 비용이 허용된다면 50~100문항으로 늘려 paired t-test 같은 간단한 통계 검정으로 차이가 유의한지 확인해 보세요.

참고: `scipy.stats.ttest_rel(naive_df["faithfulness"], adv_df["faithfulness"])`


```python
# 선택 실습: paired t-test
# 같은 질문 20개에 대해 Naive와 Advanced 점수를 짝지어 비교한다.
# p-value가 0.05보다 작으면 "차이가 우연일 가능성이 낮다"고 해석하는 경우가 많다.
# 단, 표본 20개는 작으므로 참고용으로만 보자.

from scipy.stats import ttest_rel

metric_cols = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

for col in metric_cols:
    naive_scores = naive_df_klue[col].astype(float)
    adv_scores = adv_df_klue[col].astype(float)

    stat, p_value = ttest_rel(adv_scores, naive_scores, nan_policy="omit")
    diff = (adv_scores - naive_scores).mean()

    print(f"{col}")
    print(f"  평균 차이(Advanced - Naive): {diff:.4f}")
    print(f"  t-statistic: {stat:.4f}")
    print(f"  p-value: {p_value:.4f}")
    print()
```

### 마지막 Quiz — 직접 답을 적어보세요

1. **도메인 비교**: KorQuAD(위키) 와 KLUE-MRC(뉴스) 결과에서 4지표 중 가장 크게 달라진 건 무엇이었나요? 뉴스 기사의 어떤 특성(시점 표현, 인용, 숫자 등) 때문이라고 보이나요?
2. **Advanced 효과**: KLUE-MRC 에서도 Naive → Advanced 개선폭이 컸나요? KorQuAD 와 같았나요, 달랐나요?
3. **`is_impossible` 케이스**: Step B 에서 답할 수 없는 질문을 의도적으로 섞어 평가하면 어떤 지표가 가장 망가질까요? (실험해 보면 더 좋음)
4. (선택) 같은 파이프라인을 **MIRACL ko** 로 옮기면 어떤 차이가 있을지 예상해 보세요.

## 마치며

이번 실습에서는 한국어 QA 벤치마크 위에서 다음을 진행했습니다.

- **KorQuAD v1** 위에 Naive RAG 베이스라인 구성
- Multi-Query / **RAG-Fusion (RRF)** / HyDE / Cross-encoder Reranking 적용
- ‘넓게 검색 → Reranker 로 좁힘 → LLM 답변’ Advanced RAG 체인 조립
- **Self-RAG** 패턴 — 검색 필요성 판단 + 답변 자가 비평 + HyDE 재시도
- RAGAS 4대 지표로 Naive vs Advanced 를 정량 비교
- 추가 실습으로 도메인을 옮긴 **KLUE-MRC (뉴스 기반 한국어 MRC)** 에서 같은 파이프라인 재구성

**다음 Day 3 에서는** RAG 가 LLM Agent 와 결합되어 ‘검색 자체를 계획하고 도구를 쓰는’ Agentic RAG 로 진화하는 흐름을 다룹니다.
