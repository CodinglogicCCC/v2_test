import os
from dotenv import load_dotenv
from pathlib import Path

# .env 파일 명시적으로 불러오기
dotenv_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path)
import logging
import re
from langchain_core.prompts import ChatPromptTemplate, FewShotPromptTemplate, PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from config import answer_examples  # FAQ 데이터 로드

# 로그 설정
logging.basicConfig(level=logging.INFO)

# 세션별 대화 히스토리 저장소
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """세션 ID별 대화 히스토리 관리"""
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

def get_retriever():
    """Pinecone 벡터 검색기 생성"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("❌ OPENAI_API_KEY가 설정되지 않았습니다.")

    embedding = OpenAIEmbeddings(
    model="text-embedding-3-large",
    openai_api_key=api_key,
    dimensions=1536  # Pinecone 인덱스에 맞춤
)
    index_name = "v2-test"  # 기존 Pinecone 인덱스를 유지
    database = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embedding)
    return database.as_retriever(search_kwargs={"k": 5})

def get_llm(model: str = "gpt-4o") -> ChatOpenAI:
    """GPT-4o 모델을 설정"""
    return ChatOpenAI(model=model, streaming=True)

def get_rag_chain() -> RunnableWithMessageHistory:
    """RAG 체인 생성"""
    llm = get_llm()
    retriever = get_retriever()

    system_prompt = (
        "당신은 서울과학기술대학교의 개인정보 보호 전문가입니다. "
        "검색된 문서를 반드시 활용하여 답변하세요. "
        "검색된 문서에서 가장 관련성이 높은 정보를 제공하세요. "
        "출처는 '(출처: 문서명: [문서명], 페이지: [페이지])' 형식으로 포함하세요. "
        "문서 내용과 관련 없는 정보를 생성하지 마세요. "
        "만약 검색된 문서가 부족하면 '죄송합니다. 현재 해당 질문과 관련된 정보를 찾을 수 없습니다. 다른 방식으로 질문해 보시겠어요?'라고 답변하세요.\n\n"
        "{context}"
    )

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    return RunnableWithMessageHistory(
        rag_chain, get_session_history,
        input_messages_key="input", history_messages_key="chat_history", output_messages_key="answer",
    )

# FAQ 기반 Few-shot Prompting 설정
example_template = PromptTemplate(
    input_variables=["input", "answer"],
    template="질문: {input}\n답변: {answer}\n"
)

faq_prompt = FewShotPromptTemplate(
    examples=answer_examples,
    example_prompt=example_template,
    suffix="질문: {query}\n답변:",
    input_variables=["query"]
)

def normalize_query(user_message: str) -> str:
    """사용자 입력을 표준화하여 '서울과학기술대학교'로 변환"""
    replacements = {
        "서울과기대": "서울과학기술대학교",
        "과기대": "서울과학기술대학교",
        "학교": "서울과학기술대학교"
    }
    
    for key, value in replacements.items():
        user_message = user_message.replace(key, value)
    
    logging.info(f"🛠 입력 변환됨: {user_message}")
    return user_message

def check_few_shot_examples(question: str) -> str:
    """Few-shot 데이터에서 정답이 있는지 확인"""
    for example in answer_examples:
        if example["input"] == question:
            return example["answer"]
    return None

def validate_answer_with_docs(few_shot_answer: str, retrieved_docs) -> str:
    """Few-shot 데이터와 검색된 문서가 충돌하는지 확인"""
    for doc in retrieved_docs:
        if few_shot_answer in doc.page_content:
            return few_shot_answer  # 문서와 동일하면 Few-shot 유지
    return None  # 문서와 충돌하면 검색된 문서의 내용 사용

def get_ai_response(user_message: str, session_id: str) -> str:
    """AI 응답 생성: RAG 실패 시 Few-shot Prompting을 사용"""
    try:
        # 사용자 입력을 표준화하여 "서울과학기술대학교"로 변환
        normalized_query = normalize_query(user_message)

        # Few-shot에서 먼저 답변 찾기
        answer = check_few_shot_examples(normalized_query)

        # 검색된 문서 확인
        retriever = get_retriever()
        retrieved_docs = retriever.invoke(normalized_query)

        logging.info(f"검색된 문서 개수: {len(retrieved_docs)}")
        logging.info(f"검색된 문서 내용: {retrieved_docs}")

        # Few-shot과 문서 검색 충돌 여부 확인
        if answer:
            validated_answer = validate_answer_with_docs(answer, retrieved_docs)
            if validated_answer:
                return validated_answer  # 충돌 없으면 Few-shot 유지

        if retrieved_docs:
            logging.info("검색된 문서를 기반으로 응답 생성")
            rag_chain = get_rag_chain()
            ai_response = rag_chain.invoke(
                {"input": normalized_query},
                config={"configurable": {"session_id": session_id}},
            )
        else:
            logging.info(" 검색된 문서가 없음. Few-shot Prompting 사용")

            # Few-shot Prompt 실행 로그 추가
            final_prompt = faq_prompt.format(query=normalized_query)
            logging.info(f"Few-shot Prompt 실행 프롬프트: {final_prompt}")

            llm = get_llm()
            ai_response = llm(final_prompt)

            # 실행 결과 로그 확인
            logging.info(f" Few-shot Prompt 결과: {ai_response}")

        if isinstance(ai_response, dict):
            ai_response = ai_response.get("answer", "")

        # AI 응답이 없을 경우 기본 메시지 추가
        if not ai_response or ai_response.strip() == "":
            logging.info("⚠ Few-shot Prompt 실행 후 AI 응답이 없음. 기본 응답 반환")
            return "죄송합니다. 현재 해당 질문과 관련된 정보를 찾을 수 없습니다. 더 구체적인 질문을 해주시면 도와드릴 수 있습니다."

        return ai_response.strip()

    except Exception as e:
        logging.error(f" AI 응답 생성 중 오류 발생: {e}")
        return "오류가 발생했습니다. 다시 시도해 주세요."

# 테스트 실행
retriever = get_retriever()
query = "서울과기대 라이브 방송 촬영 관리 보유 기간"  # 테스트할 질문
retrieved_docs = retriever.invoke(query)

print(f"검색된 문서 개수: {len(retrieved_docs)}")

for idx, doc in enumerate(retrieved_docs[:3]):
    print(f"\n 문서 {idx+1}: {doc.metadata}")
    print(f"내용 미리보기:\n{doc.page_content[:500]}...")