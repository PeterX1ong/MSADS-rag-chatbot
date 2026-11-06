"""
Streamlit chat UI for the MS in Applied Data Science RAG chatbot.

    streamlit run app.py

Uses the same pipeline as rag_hybrid_rerank.py: heading-aware chunks, hybrid retrieval + LLM rerank, gpt-4o-mini.
Every answer goes through guardrails.py: personal emails and phone numbers are redacted, and claims the retrieved
passages don't support are listed in a warning under the answer.
The key comes from .env locally, or from Streamlit secrets (OPENAI_API_KEY) when deployed.
"""
import os
import re

import streamlit as st
from dotenv import load_dotenv

PROMPT_VERSION = os.getenv("RAG_PROMPT", "v3")  # v3 = v1 + rules for synonyms and actions; see rag_hybrid_rerank.py
PII_HOLDBACK = 40  # chars held back while streaming, so a half-written email or phone number is never on screen
SAMPLE_QUESTIONS = [
    "How much is tuition for the program?",
    "What are the deadlines for the in-person program?",
    "What are the minimum TOEFL and IELTS scores?",
    "Is the program STEM/OPT eligible?",
]

ASSETS = os.path.join(os.path.dirname(__file__), "assets")
LOGO, ICON = os.path.join(ASSETS, "logo.svg"), os.path.join(ASSETS, "icon.svg")
PROGRAM_URL = "https://datascience.uchicago.edu/education/masters-programs/ms-in-applied-data-science/"

st.set_page_config(page_title="MS-ADS Chatbot", page_icon=ICON)
st.logo(LOGO, size="large", link=PROGRAM_URL, icon_image=ICON)

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

from guardrails import find_unsupported_claims, redact_pii
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
    parts = re.split(r"\s*\**Sources?\**:", text, maxsplit=1)  # usually on its own line, sometimes mid-paragraph
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


def show_unsupported(claims: list[str]) -> None:
    if claims:
        st.warning("Some of this answer couldn't be verified against the program website. Please double-check:\n"
                   + "\n".join(f"- {to_markdown(c)}" for c in claims))


def answer(question: str) -> dict:
    """Retrieve, stream the answer into the chat, run the guardrails, and return the message for the history."""
    retriever, generate = load_pipeline()
    with st.spinner("Searching the program website..."):
        docs = retriever.invoke(question)
    context = format_docs(docs)
    placeholder, text = st.empty(), ""
    for piece in generate.stream({"context": context, "question": question}):
        text += piece
        placeholder.markdown(to_markdown(redact_pii(split_sources(text)[0])[:-PII_HOLDBACK]))
    raw_body, sources_text = split_sources(text)
    body = redact_pii(raw_body)
    placeholder.markdown(to_markdown(body))
    sources, unsupported = [], []
    if FALLBACK not in raw_body:
        sources = cited_sources(sources_text, docs)
        with st.spinner("Checking the answer against the sources..."):
            unsupported = find_unsupported_claims(question, raw_body, context) or []
    show_sources(sources)
    show_unsupported(unsupported)
    return {"role": "assistant", "content": body, "sources": sources, "unsupported": unsupported}


with st.sidebar:
    st.markdown("### About")
    st.markdown("Ask about admissions, tuition, deadlines, courses, and visas for the UChicago "
                f"[MS in Applied Data Science]({PROGRAM_URL}).")
    st.markdown("Answers come only from the program website and cite their source pages. A warning appears when "
                "part of an answer can't be verified against those pages.")
    st.caption("Student project for ADSP 32027 Generative AI Principles. Not an official University of Chicago "
               "service. Please confirm important details with the admissions team.")

st.title("MS in Applied Data Science Chatbot")
st.caption("Answers come only from the UChicago program website. Each question is answered on its own, and "
           "important details should be confirmed with the admissions team.")

if "messages" not in st.session_state:
    st.session_state.messages = []

AVATARS = {"assistant": ICON, "user": ":material/person:"}

for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar=AVATARS[message["role"]]):
        st.markdown(to_markdown(message["content"]))
        show_sources(message.get("sources", []))
        show_unsupported(message.get("unsupported", []))

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
    with st.chat_message("user", avatar=AVATARS["user"]):
        st.markdown(to_markdown(question))
    with st.chat_message("assistant", avatar=AVATARS["assistant"]):
        try:
            st.session_state.messages.append(answer(question))
        except Exception as e:  # keep the chat usable if the API call fails
            st.session_state.messages.pop()
            st.error(f"Sorry, something went wrong ({type(e).__name__}). Please try again.")
