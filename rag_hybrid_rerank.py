"""
Heading-aware chunks (same as `heading_aware` in rag_baseline.ipynb) + hybrid retrieval + LLM rerank,
scored with the shared rag_eval.

Everything except the retriever matches rag_baseline.ipynb (data, chunking, prompt, generator, top_k), so the
leaderboard difference against `heading_aware` comes from the retriever alone.

    python rag_hybrid_rerank.py --dry-run          # answers + source_hit only, no judge calls (cents)
    python rag_hybrid_rerank.py                    # full RAGAS run, saves eval/results/hybrid_rerank.* (~$1)
    python rag_hybrid_rerank.py --no-rerank        # ablation: hybrid without the reranker -> hybrid.*
    python rag_hybrid_rerank.py --official         # the 12 Canvas questions (eval/testset_official.json) -> *_official.*
    python rag_hybrid_rerank.py --prompt v2        # revised generation prompt -> *_prompt_v2.*
"""
import argparse
import json
import os
import warnings

from dotenv import load_dotenv

warnings.filterwarnings("ignore", category=DeprecationWarning)
load_dotenv()
assert os.getenv("OPENAI_API_KEY"), "OPENAI_API_KEY not found -- add it to .env"

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

import rag_eval
import scraper
from hybrid_retriever import HybridRerankRetriever

# ── Config (same as rag_baseline.ipynb) ──────────────────────────────────────
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K = 4
EMBED_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"
RERANK_MODEL = "gpt-4o-mini"
CHROMA_DIR = "chroma_db"
COLLECTION = "msads_headings"

FALLBACK = "I don't have that information in the MS in Applied Data Science website content."


# ── Data + heading-aware chunking (copied from rag_baseline.ipynb) ───────────
def load_documents(jsonl_path=scraper.JSONL_PATH):
    docs = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            page = json.loads(line)
            docs.append(Document(page_content=page["text"], metadata={"source": page["url"], "title": page["title"]}))
    return docs


splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
HEADERS = [("#" * n, f"h{n}") for n in range(1, 7)]
header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=HEADERS, strip_headers=True)


def heading_path(doc, section):
    path = [doc.metadata["title"]] + [section.metadata[name] for _, name in HEADERS if name in section.metadata]
    return [p for i, p in enumerate(path) if p and p not in path[:i]]


def heading_aware_chunks(docs):
    chunks = []
    for doc in docs:
        children = {}
        for section in header_splitter.split_text(doc.page_content):
            path = heading_path(doc, section)
            for depth in range(1, len(path)):
                siblings = children.setdefault(tuple(path[:depth]), [])
                if path[depth] not in siblings:
                    siblings.append(path[depth])
            breadcrumb = " > ".join(path)
            meta = {**doc.metadata, "section": breadcrumb, "chunk_type": "content"}
            for piece in splitter.split_text(section.page_content):
                chunks.append(Document(page_content=f"{breadcrumb}\n\n{piece}", metadata=meta))
        for parent, kids in children.items():
            if len(kids) < 2:
                continue
            breadcrumb = " > ".join(parent)
            meta = {**doc.metadata, "section": breadcrumb, "chunk_type": "outline"}
            outline = "\n".join(f"- {k}" for k in kids)
            for piece in splitter.split_text(f"Contents of '{parent[-1]}':\n{outline}"):
                chunks.append(Document(page_content=f"{breadcrumb}\n\n{piece}", metadata=meta))
    return chunks


def build_store(chunks, collection_name: str) -> Chroma:
    embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
    Chroma(collection_name=collection_name, embedding_function=embeddings,
           persist_directory=CHROMA_DIR).delete_collection()
    return Chroma.from_documents(chunks, embeddings, collection_name=collection_name, persist_directory=CHROMA_DIR)


# ── Generation ───────────────────────────────────────────────────────────────
SYSTEM_PROMPTS = {
    # v1: the prompt from rag_baseline.ipynb
    "v1": (
        "You are a helpful assistant for the University of Chicago's MS in Applied Data Science program.\n"
        "Answer the user's question using ONLY the context below.\n"
        "- Be concise and specific.\n"
        "- End your answer with a 'Sources:' line listing the URL(s) you used.\n"
        "- If the context does not contain the answer, reply exactly: \"{fallback}\"\n\n"
        "Context:\n{context}"
    ),
    # v2: fixes the misses seen in the v1 answers -- dropped details ("Be concise"), program names that only appear
    # in headings, In-Person vs. Online differences, and refusing when the context answers part of the question.
    "v2": (
        "You are a helpful assistant for prospective students of the University of Chicago's MS in Applied Data "
        "Science program.\n"
        "Answer the user's question using ONLY the context below.\n"
        "- Be complete: include every fact in the context that answers the question, such as numbers, dates, names, "
        "locations, conditions, and exceptions. Use a short bulleted list when there are several items.\n"
        "- If the answer differs between the In-Person and Online programs, or between the 12-course program and "
        "the 18-course thesis track, give each one. If the context covers only one of them, say the other isn't "
        "covered; never assume it works the same way. If the question names one program, answer for that program.\n"
        "- Each passage starts with its heading path (for example 'Tuition, Fees, & Aid > ... > <heading>'). "
        "Headings are content: when a heading names specific scholarships, courses, or policies, list them by name.\n"
        "- Include requirements that come with the answer, such as required seminars, deposits, or documents.\n"
        "- Questions about applying (admissions counselors, appointments, deadlines, application materials) are "
        "answered by admissions information, not by profiles of academic advisors for current students.\n"
        "- If the context points the user to an action or another page (for example 'Schedule an Appointment' or a "
        "fee waiver policy), tell the user about it.\n"
        "- If the context answers only part of the question, answer that part and say what it doesn't cover.\n"
        "- Don't add facts, advice, or assumptions that aren't in the context.\n"
        "- End your answer with a 'Sources:' line listing the URL(s) you used.\n"
        "- If nothing in the context is relevant to the question, reply exactly: \"{fallback}\"\n\n"
        "Context:\n{context}"
    ),
}


def make_generator(prompt_version: str):
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPTS[prompt_version]),
        ("human", "{question}"),
    ]).partial(fallback=FALLBACK)
    return prompt | ChatOpenAI(model=LLM_MODEL, temperature=0) | StrOutputParser()


def format_docs(docs):
    return "\n\n---\n\n".join(
        f"[{i}] {d.metadata['title']} ({d.metadata['source']})\n{d.page_content}" for i, d in enumerate(docs, 1)
    )


def ask(question: str, retriever, generate, verbose: bool = True) -> dict:
    retrieved = retriever.invoke(question)
    answer = generate.invoke({"context": format_docs(retrieved), "question": question})
    if verbose:
        print(f"Q: {question}\n\nA: {answer}\n\nRetrieved chunks:")
        for i, d in enumerate(retrieved, 1):
            print(f"  [{i}] {d.metadata['title']} -- {d.metadata['source']}")
        print("=" * 70)
    return {"question": question, "answer": answer,
            "contexts": [d.page_content for d in retrieved],
            "sources": [d.metadata["source"] for d in retrieved]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="skip the RAGAS judge; print answers and source_hit")
    parser.add_argument("--no-rerank", action="store_true", help="hybrid retrieval without the LLM reranker")
    parser.add_argument("--official", action="store_true", help="evaluate on the 12 Canvas sample questions")
    parser.add_argument("--prompt", choices=SYSTEM_PROMPTS, default="v1", help="generation prompt version")
    args = parser.parse_args()

    chunks = heading_aware_chunks(load_documents())
    store = build_store(chunks, COLLECTION)
    retriever = HybridRerankRetriever(vectorstore=store, chunks=chunks, top_k=TOP_K,
                                      rerank=not args.no_rerank, rerank_model=RERANK_MODEL)
    generate = make_generator(args.prompt)
    run_name = (("hybrid" if args.no_rerank else "hybrid_rerank")
                + ("" if args.prompt == "v1" else f"_prompt_{args.prompt}")
                + ("_official" if args.official else ""))
    print(f"{len(chunks)} chunks | run: {run_name}")

    testset = rag_eval.load_testset(rag_eval.EVAL_DIR / "testset_official.json" if args.official else rag_eval.TESTSET_PATH)
    runs = rag_eval.collect(lambda q: ask(q, retriever, generate, verbose=args.dry_run), testset)

    if args.dry_run:
        in_scope = [r for r in runs if r["in_scope"]]
        hits = sum(bool(set(r["sources"]) & set(r["reference_sources"])) for r in in_scope)
        answered = sum(FALLBACK not in r["answer"] for r in in_scope)
        refused = sum(FALLBACK in r["answer"] for r in runs if not r["in_scope"])
        print(f"source_hit {hits}/{len(in_scope)} | answered {answered}/{len(in_scope)} | "
              f"out-of-scope refused {refused}/{len(runs) - len(in_scope)}")
        return

    scores = rag_eval.score(runs, fallback=FALLBACK)
    config = {"chunk_size": CHUNK_SIZE, "chunk_overlap": CHUNK_OVERLAP, "top_k": TOP_K, "embed_model": EMBED_MODEL,
              "llm_model": LLM_MODEL, "prompt": args.prompt, "chunking": "markdown_headers + breadcrumb + outline",
              "retriever": "dense + bm25 (RRF k=60, 20 candidates each)"
                           + ("" if args.no_rerank else f" + LLM rerank ({RERANK_MODEL}, pool 20)")}
    rag_eval.save_results(scores, run_name, config=config)
    print(rag_eval.leaderboard().drop(columns="evaluated_at").round(2).to_string())


if __name__ == "__main__":
    main()
