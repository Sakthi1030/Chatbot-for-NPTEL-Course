from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import streamlit as st
from dotenv import load_dotenv
import gdown
from google import genai
from google.genai import types
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from docx import Document
except ImportError:  # pragma: no cover - optional dependency fallback
    Document = None


SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".docx"}
DEFAULT_MODEL = "gemini-3-flash-preview"
DEFAULT_DOCS_DIR = "docs"


@dataclass
class Chunk:
    source: str
    text: str


def read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def read_pdf_file(path: Path) -> str:
    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def read_docx_file(path: Path) -> str:
    if Document is None:
        raise RuntimeError("python-docx is not installed. Install dependencies from requirements.txt.")
    document = Document(str(path))
    return "\n".join(paragraph.text for paragraph in document.paragraphs)


def load_document(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return read_pdf_file(path)
    if suffix in {".txt", ".md"}:
        return read_text_file(path)
    if suffix == ".docx":
        return read_docx_file(path)
    raise ValueError(f"Unsupported file type: {path.suffix}")


def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 200) -> list[str]:
    normalized = " ".join(text.split())
    if not normalized:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(normalized):
        end = start + chunk_size
        chunks.append(normalized[start:end])
        if end >= len(normalized):
            break
        start = max(end - overlap, start + 1)
    return chunks


def build_chunks(folder: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(folder.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        try:
            text = load_document(path)
        except Exception as exc:  # pragma: no cover - surfaced in UI
            st.warning(f"Skipping {path.name}: {exc}")
            continue
        for piece in chunk_text(text):
            chunks.append(Chunk(source=str(path.relative_to(folder)), text=piece))
    return chunks


def has_supported_documents(folder: Path) -> bool:
    return any(
        path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS for path in folder.rglob("*")
    )


@st.cache_resource(show_spinner=False)
def prepare_documents(folder_path: str, drive_folder_url: str) -> str:
    folder = Path(folder_path).expanduser().resolve()
    folder.mkdir(parents=True, exist_ok=True)

    if has_supported_documents(folder):
        return f"Using local documents from {folder}"

    if drive_folder_url.strip():
        gdown.download_folder(url=drive_folder_url, output=str(folder), quiet=True, remaining_ok=True)
        if has_supported_documents(folder):
            return f"Downloaded documents from Google Drive into {folder}"
        raise ValueError(
            "Google Drive download finished, but no supported documents were found. "
            "Make sure the folder is shared publicly and contains PDF, TXT, MD, or DOCX files."
        )

    return f"No documents found yet in {folder}"


def format_context(chunks: Iterable[Chunk]) -> str:
    sections: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        sections.append(f"[Source {index}: {chunk.source}]\n{chunk.text}")
    return "\n\n".join(sections)


@st.cache_resource(show_spinner=False)
def build_index(folder_path: str) -> tuple[list[Chunk], TfidfVectorizer, object]:
    folder = Path(folder_path).expanduser().resolve()
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")
    if not folder.is_dir():
        raise NotADirectoryError(f"Not a folder: {folder}")

    chunks = build_chunks(folder)
    if not chunks:
        raise ValueError(
            f"No supported documents found in {folder}. Add PDF, TXT, MD, or DOCX files."
        )

    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform([chunk.text for chunk in chunks])
    return chunks, vectorizer, matrix


def retrieve_chunks(
    query: str,
    chunks: list[Chunk],
    vectorizer: TfidfVectorizer,
    matrix: object,
    top_k: int,
) -> list[Chunk]:
    query_vector = vectorizer.transform([query])
    scores = cosine_similarity(query_vector, matrix).flatten()
    ranked_indexes = scores.argsort()[::-1]

    results: list[Chunk] = []
    for index in ranked_indexes:
        if scores[index] <= 0:
            continue
        results.append(chunks[index])
        if len(results) >= top_k:
            break
    return results


def generate_answer(api_key: str, model_name: str, query: str, context: str) -> str:
    client = genai.Client(api_key=api_key)
    prompt = f"""
You are a helpful course assistant.
Answer the question only from the provided context.
If the answer is not in the context, say that clearly.
Keep the answer concise and cite the source labels when possible.

Question:
{query}

Context:
{context}
""".strip()

    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.2,
            max_output_tokens=700,
        ),
    )
    return response.text or "No response was returned by the model."


def render_sidebar() -> tuple[str, str, int]:
    st.sidebar.header("Settings")

    api_key = st.sidebar.text_input(
        "Gemini API key",
        value=os.getenv("GEMINI_API_KEY", ""),
        type="password",
        help="Create one in Google AI Studio and store it in a .env file if you want.",
    )
    model_name = st.sidebar.text_input(
        "Model",
        value=os.getenv("GEMINI_MODEL", DEFAULT_MODEL),
        help="Default is gemini-3-flash-preview. If availability changes, try gemini-2.5-flash.",
    )
    top_k = st.sidebar.slider("Retrieved chunks", min_value=2, max_value=8, value=4)

    st.sidebar.caption(
        "Supported files: PDF, TXT, MD, DOCX. Point the app at a folder that mirrors your Drive content."
    )
    return api_key, model_name, top_k


def main() -> None:
    load_dotenv()
    st.set_page_config(page_title="NPTEL RAG Chatbot", page_icon=":books:", layout="wide")
    st.title("NPTEL Course RAG Chatbot")
    st.write(
        "Ask questions over course documents stored in a local folder, synced Drive folder, or mounted Drive path."
    )

    api_key, model_name, top_k = render_sidebar()

    default_docs_path = os.getenv("DOCS_DIR", DEFAULT_DOCS_DIR)
    drive_folder_url = os.getenv("GDRIVE_FOLDER_URL", "")
    docs_path = st.text_input(
        "Document folder path",
        value=default_docs_path,
        placeholder=r"C:\Users\you\Google Drive\NPT file",
        help="Use any folder that contains your course documents.",
    )
    if drive_folder_url:
        st.caption("A Google Drive folder URL is configured for automatic document download.")

    if not docs_path:
        st.info("Enter a document folder path to build the retriever.")
        return

    try:
        prep_message = prepare_documents(docs_path, drive_folder_url)
        st.caption(prep_message)
    except Exception as exc:
        st.error(f"Document preparation failed: {exc}")
        return

    try:
        chunks, vectorizer, matrix = build_index(docs_path)
    except Exception as exc:
        st.error(str(exc))
        return

    st.success(f"Indexed {len(chunks)} chunks from {docs_path}")

    question = st.text_input("Ask a question about your documents")
    ask_clicked = st.button("Ask")

    if ask_clicked:
        if not question.strip():
            st.warning("Enter a question first.")
            return
        if not api_key.strip():
            st.warning("Add your Gemini API key in the sidebar first.")
            return

        matches = retrieve_chunks(question, chunks, vectorizer, matrix, top_k=top_k)
        if not matches:
            st.warning("I could not find relevant context in the indexed documents.")
            return

        context = format_context(matches)
        with st.spinner("Generating answer..."):
            try:
                answer = generate_answer(api_key, model_name, question, context)
            except Exception as exc:
                st.error(f"Model request failed: {exc}")
                return

        st.subheader("Answer")
        st.write(answer)

        with st.expander("Retrieved context"):
            for index, chunk in enumerate(matches, start=1):
                st.markdown(f"**Source {index}:** `{chunk.source}`")
                st.write(chunk.text)


if __name__ == "__main__":
    main()
