# df_rag.py
from __future__ import annotations
from typing import List, Optional
import pandas as pd

from langchain.docstore.document import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import SystemMessage, HumanMessage

# ---------- DF -> Documents ----------
def df_to_documents(
    df: pd.DataFrame,
    use_cols: Optional[List[str]] = None,
    id_col: Optional[str] = None,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> List[Document]:
    """
    각 행(row)을 한 개 문서로 만들고, 길면 청크로 분할합니다.
    - use_cols: RAG에 사용할 컬럼만 선택 (None이면 전 컬럼)
    - id_col: 행 식별자로 메타데이터에 포함
    """
    if use_cols:
        df = df[use_cols]

    docs: List[Document] = []
    for idx, row in df.iterrows():
        # 한 행을 "col: value" 형태로 직렬화
        lines = [f"{col}: {row[col]}" for col in df.columns]
        text = "\n".join(lines)

        meta = {"row_index": int(idx)}
        if id_col and id_col in df.columns:
            meta["id"] = str(row[id_col])

        docs.append(Document(page_content=text, metadata=meta))

    # 너무 길면 청크
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    chunked = splitter.split_documents(docs)
    return chunked

# ---------- VectorStore ----------
def build_vectorstore(
    docs: List[Document],
    embeddings: Optional[OpenAIEmbeddings] = None
) -> FAISS:
    embeddings = embeddings or OpenAIEmbeddings()
    vs = FAISS.from_documents(docs, embeddings)
    return vs

def retrieve_context(
    query: str,
    vs: FAISS,
    k: int = 15
) -> tuple[str, List[Document]]:
    """쿼리와 가장 유사한 상위 k개 문서를 찾아 텍스트로 합칩니다."""
    docs = vs.similarity_search(query, k=k)
    context = "\n\n---\n\n".join(d.page_content for d in docs)
    return context, docs

# ---------- LLM 호출 ----------
def ask_llm_with_context(
    question: str,
    context: str,
    model: str = "gpt-5-mini",
    temperature: float = 0.0
) -> str:
    sys = SystemMessage(content=(
        "너는 표/데이터프레임 기반 RAG 어시스턴트야. "
        "아래 CONTEXT에서 근거를 찾아 한국어로 간결히 답해. "
        "근거가 없으면 모른다고 말해."
    ))
    user = HumanMessage(content=[
        {"type": "text", "text":
         f"QUESTION:\n{question}\n\nCONTEXT:\n{context}\n\n"
         "규칙: 망상 금지, 컨텍스트 외 추정 금지."}
    ])
    llm = ChatOpenAI(model=model, temperature=temperature)
    resp = llm.invoke([sys, user])
    return resp.content
