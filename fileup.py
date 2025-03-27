import os
import logging
from dotenv import load_dotenv
from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

# .env 파일 로드
dotenv_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path)

# 로그 설정
logging.basicConfig(level=logging.INFO)

# 설정 (최종)
INDEX_NAME = "v2-test"
DATA_DIR = r"C:\Users\sukyo\privacychat\DB\v2 마크다운_표 했을떄"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

def load_documents():
    """단일 폴더 내 .md 문서들을 직접 로드 (TextLoader 사용)"""
    all_docs = []
    for root, _, files in os.walk(DATA_DIR):
        for file in files:
            if file.endswith(".md"):
                file_path = os.path.join(root, file)
                try:
                    loader = TextLoader(file_path, encoding="utf-8")
                    docs = loader.load()
                    all_docs.extend(docs)
                    logging.info(f"로드됨: {file_path}")
                except Exception as e:
                    logging.warning(f" 파일 로드 실패: {file_path} ({e})")
    logging.info(f"총 문서 수: {len(all_docs)}")
    return all_docs

def split_documents(documents):
    """문서 청크 분할"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    return splitter.split_documents(documents)

def embed_and_upload(docs):
    """문서를 임베딩하고 Pinecone에 배치로 업로드"""
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        raise ValueError("OPENAI_API_KEY가 .env에 설정되지 않았습니다.")

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large",
        dimensions=1536,  # 중요: Pinecone 인덱스 차원과 맞춰줌
        openai_api_key=openai_key
    )

    logging.info(f" 전체 문서 청크 수: {len(docs)}")

    # 배치 크기 설정
    batch_size = 100

    for i in range(0, len(docs), batch_size):
        batch = docs[i:i + batch_size]
        logging.info(f"Pinecone에 문서 업로드 중... ({i} ~ {i + len(batch)})")
        PineconeVectorStore.from_documents(
            batch,
            embedding=embeddings,
            index_name=INDEX_NAME
        )

    logging.info("전체 문서 업로드 완료!")

def main():
    logging.info("문서 업로드 시작")
    raw_docs = load_documents()
    split_docs = split_documents(raw_docs)
    embed_and_upload(split_docs)

if __name__ == "__main__":
    main()
