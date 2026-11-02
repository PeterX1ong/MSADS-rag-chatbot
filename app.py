"""
Streamlit chat UI for the MS in Applied Data Science RAG chatbot.

    streamlit run app.py

Uses the same pipeline as rag_hybrid_rerank.py: heading-aware chunks, hybrid retrieval + LLM rerank, gpt-4o-mini.
The key comes from .env locally, or from Streamlit secrets (OPENAI_API_KEY) when deployed.
"""
import os
import re

import streamlit as st
from dotenv import load_dotenv

PROMPT_VERSION = os.getenv("RAG_PROMPT", "v1")  # v1 scored best on RAGAS; see eval/results/
SAMPLE_QUESTIONS = [
    "How much is tuition for the program?",
    "What are the deadlines for the in-person program?",
    "What are the minimum TOEFL and IELTS scores?",
    "Is the program STEM/OPT eligible?",
]

st.set_page_config(page_title="MS-ADS Chatbot", page_icon="🎓")

load_dotenv()
try:
    os.environ.setdefault("OPENAI_API_KEY", st.secrets["OPENAI_API_KEY"])
except (KeyError, FileNotFoundError, st.errors.StreamlitSecretNotFoundError):
    pass  # no secrets file: running locally on .env
if not os.getenv("OPENAI_API_KEY"):
    st.error("OPENAI_API_KEY is not set. Add it to .env, or to Streamlit secrets when deployed.")
    st.stop()

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from hybrid_retriever import HybridRerankRetriever
from rag_hybrid_rerank import (CHROMA_DIR, COLLECTION, EMBED_MODEL, FALLBACK, RERANK_MODEL, TOP_K, build_store,
                               format_docs, heading_aware_chunks, load_documents, make_generator)


@st.cache_resource(show_spinner="Loading the program website index...")
def load_pipeline():
    """Reuse the persisted vector store when it matches the chunks; embed only on the first run."""
    chunks = heading_aware_chunks(load_documents())
    store = Chroma(collection_name=COLLECTION, embedding_function=OpenAIEmbeddings(model=EMBED_MODEL),
                   persist_directory=CHROMA_DIR)
    if store._collection.count() != len(chunks):
        store = build_store(chunks, COLLECTION)
    retriever = HybridRerankRetriever(vectorstore=store, chunks=chunks, top_k=TOP_K, rerank_model=RERANK_MODEL)
    return retriever, make_generator(PROMPT_VERSION)


def split_sources(text: str) -> tuple[str, str]:
    """Split a generated answer into its body and the trailing 'Sources:' section."""
    parts = re.split(r"\n\s*\**Sources?\**:", text, maxsplit=1)
    return parts[0].strip(), parts[1] if len(parts) > 1 else ""


def to_markdown(text: str) -> str:
    """Escape the characters Streamlit would render as LaTeX ($6,384 ... $76,608) or strikethrough (~50)."""
    return text.replace("$", r"\$").replace("~", r"\~")


def cited_sources(sources_text: str, docs) -> list[tuple[str, str]]:
    """(title, url) for each URL the answer cites. Only retrieved pages count, so a made-up link can't be shown."""
    titles = {d.metadata["source"]: d.metadata["title"] for d in docs}
    cited = dict.fromkeys(re.findall(r"https?://[^\s)\]>]+", sources_text))
    return [(titles[url], url) for url in cited if url in titles]


def show_sources(sources: list[tuple[str, str]]) -> None:
    if sources:
        st.caption("Sources: " + " · ".join(f"[{title}]({url})" for title, url in sources))


def answer(question: str) -> dict:
    """Retrieve, stream the answer into the chat, and return the message to keep in the history."""
    retriever, generate = load_pipeline()
    with st.spinner("Searching the program website..."):
        docs = retriever.invoke(question)
    placeholder, text = st.empty(), ""
    for piece in generate.stream({"context": format_docs(docs), "question": question}):
        text += piece
        placeholder.markdown(to_markdown(split_sources(text)[0]))
    body, sources_text = split_sources(text)
    sources = [] if FALLBACK in body else cited_sources(sources_text, docs)
    show_sources(sources)
    return {"role": "assistant", "content": body, "sources": sources}


st.title("MS in Applied Data Science Chatbot")
st.caption("Answers come only from the UChicago program website. Each question is answered on its own, and "
           "important details should be confirmed with the admissions team.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(to_markdown(message["content"]))
        show_sources(message.get("sources", []))

question = st.chat_input("Ask about admissions, tuition, courses...")
samples = st.empty()
if not st.session_state.messages and not question:
    with samples.container():
        for column, sample in zip(st.columns(2) * 2, SAMPLE_QUESTIONS):
            if column.button(sample, use_container_width=True):
                question = sample

if question:
    samples.empty()
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(to_markdown(question))
    with st.chat_message("assistant"):
        try:
            st.session_state.messages.append(answer(question))
        except Exception as e:  # keep the chat usable if the API call fails
            st.session_state.messages.pop()
            st.error(f"Sorry, something went wrong ({type(e).__name__}). Please try again.")
