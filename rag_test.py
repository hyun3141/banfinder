import os
import re
import io
import zipfile
import subprocess
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# (선택) 가용 시 사용
try:
    import olefile  # .hwp OLE 컨테이너 접근용
except Exception:
    olefile = None

# PDF 추출
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer, LTTextBox, LTTextLine


# ---------------------------
# 텍스트 추출 유틸
# ---------------------------

def extract_text_pdf_per_page(path: str) -> List[Tuple[int, str]]:
    """PDF에서 페이지별 텍스트 추출"""
    pages = []
    try:
        for i, layout in enumerate(extract_pages(path), start=1):
            texts = []
            for el in layout:
                if isinstance(el, (LTTextContainer, LTTextBox, LTTextLine)):
                    try:
                        texts.append(el.get_text())
                    except Exception:
                        pass
            page_text = "".join(texts).strip()
            if page_text:
                pages.append((i, page_text))
    except Exception as e:
        st.warning(f"[PDF 읽기 실패] {path} - {e}")
    return pages


def extract_text_hwpx(path: str) -> str:
    """
    HWPX(압축된 XML 패키지)에서 텍스트 추출(범용).
    구조를 깊이 해석하지 않고 모든 XML 텍스트 노드를 수집.
    """
    text_nodes = []
    try:
        with zipfile.ZipFile(path, 'r') as zf:
            # 텍스트 본문이 들어있는 섹션/본문 XML들을 우선
            candidates = [n for n in zf.namelist() if n.lower().endswith(".xml")]
            for name in candidates:
                try:
                    data = zf.read(name)
                    # lxml이 있으면 사용(속도↑), 없으면 표준 lib로 처리
                    try:
                        from lxml import etree as ET  # type: ignore
                    except Exception:
                        import xml.etree.ElementTree as ET  # fallback
                    root = ET.fromstring(data)
                    # 모든 텍스트 노드를 수집
                    for elem in root.iter():
                        t = (elem.text or "").strip()
                        if t:
                            text_nodes.append(t)
                except Exception:
                    continue
    except Exception as e:
        st.warning(f"[HWPX 읽기 실패] {path} - {e}")
    return "\n".join(text_nodes).strip()


def extract_text_hwp_via_hwp5txt(path: str) -> Optional[str]:
    """
    hwp5txt CLI가 있으면 그것으로 .hwp 텍스트 추출.
    리눅스 배포판(hwp5-tools)에서 주로 제공. 없으면 None.
    """
    try:
        out = subprocess.check_output(["hwp5txt", path], stderr=subprocess.STDOUT, timeout=20)
        return out.decode("utf-8", errors="ignore")
    except Exception:
        return None


def extract_text_hwp_best_effort(path: str) -> str:
    """
    .hwp(5.x) 베스트에포트 추출:
    1) hwp5txt 사용 시도
    2) olefile로 BodyText/Section* 스트림 긁어서 가시 텍스트 후보 추출(완벽 X)
    """
    # 1) hwp5txt가 있으면 우선 사용
    t = extract_text_hwp_via_hwp5txt(path)
    if isinstance(t, str) and t.strip():
        return t

    # 2) olefile로 원시 바이트에서 한글/영문 가시문자만 추출하는 간이 버전
    if olefile is None:
        st.info(f"[안내] .hwp 추출을 원활히 하려면 'hwp5txt' 설치 또는 'olefile' 모듈 설치를 권장합니다: {path}")
        return ""

    try:
        of = olefile.OleFileIO(path)
        # BodyText 안의 Section* 스트림들 대상
        sections = [s for s in of.listdir() if len(s) >= 2 and s[0] == 'BodyText' and s[1].startswith('Section')]
        texts = []
        for sect in sections:
            try:
                data = of.openstream(sect).read()
                # 압축/포맷을 완벽히 해석하진 않고, 가시 문자열 힌트만 추출(임시)
                # 한글/영문/숫자/공백/구두점 등을 뽑아내는 정규식
                guess = re.findall(r"[ \t\r\nA-Za-z0-9가-힣ㄱ-ㅎㅏ-ㅣ.,:;!?()\-_/\\\[\]{}\"'`~@#$%^&*=+<>|]+", data)
                chunk = b" ".join(guess).decode("utf-8", errors="ignore")
                # 비정상 반복 공백 축소
                chunk = re.sub(r"\s{2,}", " ", chunk)
                texts.append(chunk)
            except Exception:
                continue
        of.close()
        return "\n".join(texts).strip()
    except Exception as e:
        st.warning(f"[HWP 읽기 실패] {path} - {e}")
        return ""


def read_file_to_units(path: str) -> List[Tuple[str, Optional[int]]]:
    """
    파일을 '검색 가능한 단위(text, page)' 리스트로 변환.
    - PDF: 페이지별 단위
    - HWP/HWPX: 문서 전체(페이지 정보 불명) 1개 단위
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        pages = extract_text_pdf_per_page(path)
        return [(txt, pno) for (pno, txt) in pages if txt.strip()]
    elif ext == ".hwpx":
        txt = extract_text_hwpx(path)
        return [(txt, None)] if txt.strip() else []
    elif ext == ".hwp":
        txt = extract_text_hwp_best_effort(path)
        return [(txt, None)] if txt.strip() else []
    else:
        return []


# ---------------------------
# 색인/검색
# ---------------------------

@dataclass
class Passage:
    doc_path: str
    page: Optional[int]
    text: str


def split_into_chunks(text: str, chunk_size: int = 800, overlap: int = 150) -> List[str]:
    """
    문단 기준으로 합치되, 최종적으로는 문자 길이로 슬라이싱.
    한국어에서는 char n-gram 기반 검색이 잘 동작하므로 문자 단위가 안전.
    """
    text = re.sub(r"\r\n|\r", "\n", text)
    paras = [p.strip() for p in text.split("\n") if p.strip()]
    buf, chunks = "", []
    for p in paras:
        # 새 문단 추가 시 길이 체크
        if len(buf) + len(p) + 1 <= chunk_size:
            buf = (buf + "\n" + p).strip() if buf else p
        else:
            if buf:
                chunks.append(buf)
            # 오버랩 적용
            if overlap > 0 and chunks:
                tail = chunks[-1][-overlap:]
                buf = (tail + "\n" + p).strip()
            else:
                buf = p
    if buf:
        chunks.append(buf)

    # 혹시 너무 긴 단락이 들어왔을 때 추가 슬라이스
    final = []
    for c in chunks:
        if len(c) <= chunk_size:
            final.append(c)
        else:
            i = 0
            while i < len(c):
                final.append(c[i:i+chunk_size])
                i += (chunk_size - overlap) if overlap < chunk_size else chunk_size
    return final


def build_index(root_dir: str, exts: Tuple[str, ...] = (".pdf", ".hwp", ".hwpx"),
                chunk_size: int = 800, overlap: int = 150) -> Tuple[List[Passage], TfidfVectorizer]:
    passages: List[Passage] = []
    # 파일 수집
    file_list = []
    for r, _, files in os.walk(root_dir):
        for fn in files:
            if os.path.splitext(fn)[1].lower() in exts:
                file_list.append(os.path.join(r, fn))

    progress = st.progress(0.0, text="문서를 읽는 중...")
    total = max(1, len(file_list))

    for idx, path in enumerate(sorted(file_list)):
        units = read_file_to_units(path)
        for (txt, page) in units:
            if not txt or len(txt.strip()) == 0:
                continue
            for c in split_into_chunks(txt, chunk_size=chunk_size, overlap=overlap):
                passages.append(Passage(doc_path=path, page=page, text=c))
        progress.progress((idx + 1) / total, text=f"색인 중... ({idx+1}/{total})")

    progress.empty()

    # 한국어 친화: char n-gram 기반 TF-IDF
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=1,
        max_df=1.0
    )
    corpus = [p.text for p in passages] if passages else ["(empty)"]
    X = vectorizer.fit_transform(corpus)

    # 세션 상태에 저장
    st.session_state["index_matrix"] = X
    st.session_state["vectorizer"] = vectorizer
    st.session_state["passages"] = passages
    st.session_state["root_dir"] = root_dir

    return passages, vectorizer


def ensure_index(root_dir: str, chunk_size: int = 1000, overlap: int = 120) -> bool:
    """세션 상태에 인덱스가 없거나 경로/파라미터가 바뀌면 새로 생성"""
    need_rebuild = False
    if "passages" not in st.session_state:
        need_rebuild = True
    else:
        if st.session_state.get("root_dir") != root_dir:
            need_rebuild = True
        if st.session_state.get("chunk_size") != chunk_size or st.session_state.get("overlap") != overlap:
            need_rebuild = True

    if need_rebuild:
        passages, _ = build_index(root_dir)
        st.session_state["chunk_size"] = chunk_size
        st.session_state["overlap"] = overlap
        if not passages:
            st.warning("인덱스에 추가된 문서가 없습니다. 폴더/확장자를 확인하세요.")
            return False
    return True


def search(query: str, top_k: int = 10) -> List[Tuple[float, Passage]]:
    vectorizer: TfidfVectorizer = st.session_state["vectorizer"]
    X = st.session_state["index_matrix"]
    passages: List[Passage] = st.session_state["passages"]

    qv = vectorizer.transform([query])
    sims = linear_kernel(qv, X).flatten()  # cosine 유사도
    # top_k 인덱스 추출
    top_idx = sims.argsort()[::-1][:top_k]
    results = [(float(sims[i]), passages[i]) for i in top_idx]
    return results


def highlight(text: str, query: str) -> str:
    """간단 하이라이트(대소문자 무시). 한국어도 부분 일치."""
    if not query.strip():
        return text
    try:
        pattern = re.escape(query.strip())
        return re.sub(pattern, lambda m: f"**{m.group(0)}**", text, flags=re.IGNORECASE)
    except Exception:
        return text


# ---------------------------
# UI
# ---------------------------

st.set_page_config(page_title="문서 검색 (PDF/HWP/HWPX) — LLM 없는 RAG", layout="wide")

st.title("🔎 문서 검색 (PDF/HWP/HWPX")
st.caption("폴더 내 문서를 색인하고 TF-IDF로 검색합니다. 결과에는 근거(파일명/페이지)를 함께 표시합니다.")

root_dir = os.getcwd() + r"\rag용파일"
chunk_size = 800
overlap = 150
top_k = 5


# 처음 진입/재색인
if "passages" not in st.session_state:
    ensure_index(root_dir, chunk_size, overlap)


# 검색 영역
query = st.text_input("검색어를 입력하세요", value="", placeholder="예) 전기안전 점검 주기, 지연 배상 기준, 계약 해지 조항 ...")
search_btn = st.button("검색 실행")

if search_btn and query.strip():
    if "passages" not in st.session_state:
        st.warning("먼저 색인을 생성해주세요.")
    else:
        results = search(query, top_k=top_k)
        if not results:
            st.info("검색 결과가 없습니다.")
        else:
            st.write(f"**검색 결과 {len(results)}건**")
            for score, passage in results:
                src = os.path.relpath(passage.doc_path, start=root_dir) if st.session_state.get("root_dir") else passage.doc_path
                where = f"{src}" + (f" · p.{passage.page}" if passage.page else "")
                with st.container(border=True):
                    st.markdown(f"**근거:** `{where}`  |  유사도: `{score:.3f}`")
                    st.markdown(highlight(passage.text, query))

# 하단 안내
with st.expander("ℹ️ .hwp 지원 관련 안내"):
    st.markdown(
        "- 최적: 시스템에 `hwp5txt`(hwp5-tools)가 설치되어 있으면 고품질 텍스트 추출을 시도합니다.\n"
        "- 대안: `olefile`만 있는 경우 내부 스트림에서 가시 텍스트를 추정 추출(완벽하지 않을 수 있음).\n"
        "- 정확도가 중요한 .hwp는 **HWPX로 저장**하거나 **PDF로 변환** 후 색인을 권장합니다."
    )


