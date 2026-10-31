"""
Hybrid retrieval with LLM reranking, as a drop-in LangChain retriever for `ask(question, retriever=...)`.

    dense (Chroma, text-embedding-3-small) ─┐
                                            ├─ Reciprocal Rank Fusion ─> LLM rerank (0-3 per chunk) ─> top_k
    sparse (BM25 over the same chunks)    ──┘

- Dense search matches paraphrases ("cost" ~ "tuition").
- BM25 matches exact terms the embedding underweights ("TOEFL", "MBA/MS", "enrollment management", "fee waiver").
- RRF merges the two rankings by rank, so their score scales never need calibrating.
- The reranker reads every fused candidate and scores how directly it answers the question. This fixes chunks that
  share the question's wording but not its meaning (current-student advising vs. admissions counselors, the MSDS
  program's "9 courses" vs. MS-ADS's 12, Online vs. In-Person rules).
"""
import json
import re

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore
from openai import OpenAI
from rank_bm25 import BM25Okapi

RERANK_PROMPT = """You rank search results for a Q&A assistant about UChicago's MS in Applied Data Science program.

Question: {question}

Score every passage for how well it helps answer the question:
3 = directly contains the answer
2 = partially answers or gives key supporting facts
1 = same topic but does not answer
0 = unrelated, or about a different audience/program than the question asks about

Passages:
{passages}

Return JSON only: {{"scores": {{"<passage id>": <0-3>, ...}}}} with an entry for every passage id."""

STOPWORDS = set("""a an the and or of to in for on at by with is are was be can do does i my me you your
what which how when where who will it this that from as program programs""".split())


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens for BM25; keeps slashed terms like "mba/ms" and "stem/opt" as one token."""
    return [t for t in re.findall(r"[a-z0-9]+(?:/[a-z0-9]+)?", text.lower()) if t not in STOPWORDS]


class HybridRerankRetriever(BaseRetriever):
    """Dense + BM25 fused with RRF, then reranked by a small LLM. `chunks` must be the documents in `vectorstore`."""

    vectorstore: VectorStore
    chunks: list[Document]
    top_k: int = 4
    candidates: int = 20      # per retriever, before fusion
    rrf_k: int = 60
    rerank: bool = True       # False = hybrid only (useful as an ablation)
    rerank_pool: int = 20     # fused candidates the reranker reads
    rerank_model: str = "gpt-4o-mini"
    snippet_chars: int = 900  # chars of each candidate shown to the reranker

    _bm25: BM25Okapi
    _client: OpenAI

    def model_post_init(self, __context) -> None:
        self._bm25 = BM25Okapi([tokenize(d.page_content) for d in self.chunks])
        self._client = OpenAI()

    def _dense(self, query: str) -> list[int]:
        """Indices into self.chunks, matched back by content since Chroma doesn't return our list positions."""
        position = {d.page_content: i for i, d in enumerate(self.chunks)}
        hits = self.vectorstore.similarity_search(query, k=self.candidates)
        return [position[d.page_content] for d in hits if d.page_content in position]

    def _sparse(self, query: str) -> list[int]:
        scores = self._bm25.get_scores(tokenize(query))
        order = sorted(range(len(self.chunks)), key=lambda i: scores[i], reverse=True)
        return [i for i in order[:self.candidates] if scores[i] > 0]

    def _hybrid(self, query: str) -> list[int]:
        fused: dict[int, float] = {}
        for ranking in (self._dense(query), self._sparse(query)):
            for rank, i in enumerate(ranking):
                fused[i] = fused.get(i, 0.0) + 1 / (self.rrf_k + rank + 1)
        return sorted(fused, key=fused.get, reverse=True)

    def _rerank(self, query: str, ids: list[int]) -> list[int]:
        passages = "\n\n".join(
            f'<passage id="{i}">\n{self.chunks[i].page_content[:self.snippet_chars]}\n</passage>' for i in ids
        )
        resp = self._client.chat.completions.create(
            model=self.rerank_model,
            temperature=0,
            messages=[{"role": "user", "content": RERANK_PROMPT.format(question=query, passages=passages)}],
            response_format={"type": "json_object"},
        )
        try:
            scores = json.loads(resp.choices[0].message.content)["scores"]
        except (json.JSONDecodeError, KeyError, TypeError):
            return ids  # fall back to the fused order rather than failing the question
        # Stable sort: chunks with equal LLM scores keep their hybrid order.
        return sorted(ids, key=lambda i: -float(scores.get(str(i), 0)))

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        ranked = self._hybrid(query)
        if self.rerank:
            ranked = self._rerank(query, ranked[:self.rerank_pool])
        return [self.chunks[i] for i in ranked[:self.top_k]]
