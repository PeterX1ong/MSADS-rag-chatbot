# MS in Applied Data Science RAG Chatbot

ADSP 32027 Generative AI Principles, Course Project 1.

This is a retrieval-augmented generation (RAG) chatbot that answers questions about the
[UChicago MS in Applied Data Science](https://datascience.uchicago.edu/education/masters-programs/ms-in-applied-data-science/)
program using only the program's website. Answers cite the source pages. When the site doesn't contain the answer,
the chatbot says so instead of guessing.

**Live app:** https://msads-rag-chatbot.streamlit.app/ (see [Deployment](#deployment)).

`main` holds the **baseline** pipeline and a shared **RAGAS evaluation**. Each teammate builds an alternative approach
on their own branch and scores it with the same evaluation, so we can compare approaches and pick the best.

## Repository layout

| Path | What it is |
|---|---|
| `scraper.py` | Crawls the program site (main page, sublinks, faculty profiles, MS-ADS news) into `data/` |
| `data/pages.jsonl` | Scraped pages, one JSON record per page (`url`, `title`, `text`, ...). Committed so everyone uses the same data |
| `data/raw_pages/*.txt` | The same pages as readable text files |
| `rag_baseline.ipynb` | End-to-end pipeline: scrape, chunk, embed into Chroma, answer with GPT, evaluate |
| `hybrid_retriever.py` | `HybridRerankRetriever`: dense + BM25 fused with RRF, then an LLM rerank. Drop-in LangChain retriever |
| `rag_hybrid_rerank.py` | Heading-aware pipeline with the hybrid retriever, as a script: answers, RAGAS scoring, prompt versions |
| `guardrails.py` | Responsible-AI checks on every answer: personal email/phone redaction and an LLM groundedness check |
| `app.py` | Streamlit chat UI on the hybrid + rerank pipeline, with the guardrails applied |
| `.streamlit/config.toml`, `assets/` | UI theme (UChicago maroon) and the app's own logo. Keys never go in `config.toml` |
| `rag_eval.py` | Shared evaluation module (RAGAS metrics, saving results, leaderboard) |
| `eval/testset.json` | Shared test set: 15 in-scope questions with hand-written reference answers, plus 2 out-of-scope questions |
| `eval/testset_official.json` | The 12 Canvas sample questions DSI staff use to test the bot. Separate from the shared test set |
| `eval/results/` | One `<run_name>.json` (summary) and `.csv` (per question) file per evaluated approach |

## Setup

1. Install the dependencies (tested with Python 3.13):
   ```bash
   pip install -r requirements.txt
   ```
   Keep the pinned versions. `ragas` must stay on **0.3.1**: newer releases downgrade `langchain-openai` and `openai`.
2. Add your OpenAI API key:
   ```bash
   cp .env.example .env    # then paste your key into .env
   ```
   `.env` is git-ignored. Never commit a key.
3. Open `rag_baseline.ipynb` and run all cells.
   - `RUN_SCRAPER = False` (the default) uses the committed `data/pages.jsonl`. Keep it that way when comparing
     approaches so everyone works from the same data. Set it to `True` only to re-crawl the site, which takes about
     2 minutes.
   - `chroma_db/` is rebuilt automatically and isn't committed.
4. Start the chatbot:
   ```bash
   streamlit run app.py
   ```
   The first run embeds the chunks if `chroma_db/` doesn't match them. The app uses prompt `v3` by default;
   `RAG_PROMPT=v1` or `v2` switches it.

## Deployment

The app runs on [Streamlit Community Cloud](https://share.streamlit.io) at
https://msads-rag-chatbot.streamlit.app/.

| Setting | Value |
|---|---|
| Repository | `PeterX1ong/MSADS-rag-chatbot` (a copy of this repo, since the cloud host needs access to it) |
| Branch / main file | `PeterX` / `app.py` |
| Python | **3.13**. The default 3.14 fails to build the pinned `lxml==5.3.0`, and the version can only be changed by deleting and redeploying the app |
| Secrets | `OPENAI_API_KEY = "sk-..."` under the app's Settings → Secrets. Never in the code or `config.toml` |

**Updating the live app.** Streamlit Cloud watches the deploy branch and redeploys on every push, usually within a
minute or two. A local clone has two remotes:

```bash
git remote add mine https://github.com/PeterX1ong/MSADS-rag-chatbot.git   # once
git push origin PeterX    # team repo
git push mine PeterX      # deploy repo: this is what updates the live app
```

| Change | What happens |
|---|---|
| Code, prompts, README | Picked up automatically after the push |
| `requirements.txt` | Dependencies reinstall, which takes a few minutes. Failures show "Error installing requirements" |
| `.streamlit/config.toml` | Usually picked up automatically; otherwise use **Reboot app** |
| API key | Change it in Settings → Secrets (the app restarts). The local `.env` is separate, so update both |
| Python version | Delete the app and deploy it again |

Notes:

- Logs are under **Manage app** (bottom right of the live app). That's the first place to look when it fails.
- `chroma_db/` isn't committed, so every start re-embeds the 718 chunks (about a minute, under $0.01).
- Free apps sleep after a period without visitors. Open the app once before a demo or grading so it's awake.
- Every question costs a little on the OpenAI key in Secrets. Set a monthly spending limit on that key.

## Evaluation rules (please read before comparing approaches)

For scores to be comparable across branches, everyone must use **the same judge, test set, and data**.

| Setting | Value | Where it's set |
|---|---|---|
| Judge LLM | **`gpt-4o`** (temperature 0) | `JUDGE_MODEL` in `rag_eval.py` |
| Judge embeddings | `text-embedding-3-small` | `JUDGE_EMBED_MODEL` in `rag_eval.py` |
| Test set | `eval/testset.json` (17 questions) | `TESTSET_PATH` in `rag_eval.py` |
| Corpus | committed `data/pages.jsonl` | `RUN_SCRAPER = False` |

- **Always use `gpt-4o` as the judge.** It is deliberately stronger than, and different from, the `gpt-4o-mini`
  generator, so the model isn't grading its own answers.
- **Don't change the judge, test set, or metrics on a feature branch.** If we want to change them, we do it on `main`
  together and re-score every approach.
- A run costs roughly $1 in judge calls per approach. The actual cost is printed after each run.
- LLM judges aren't perfectly deterministic. Treat differences of a few points as noise.

### Metrics

All metrics are averaged over in-scope questions, range from 0 to 1, and are better when higher.

| Metric | Stage | Question it answers |
|---|---|---|
| `context_precision` | retrieval | Are the chunks that support the reference answer ranked at the top? |
| `context_recall` | retrieval | Do the retrieved chunks contain everything the reference answer says? |
| `source_hit` | retrieval | Did at least one retrieved chunk come from a page the answer is on? (no LLM) |
| `faithfulness` | generation | Is every claim in the answer supported by the retrieved chunks? (hallucination check) |
| `answer_relevancy` | generation | Does the answer address the question that was asked? |
| `factual_correctness` | end-to-end | Claim-level F1 score of the answer compared with the reference answer |
| `answered` | end-to-end | Share of in-scope questions answered instead of getting the fallback message |
| `out_of_scope_refusal` | guardrail | Share of out-of-scope questions that correctly got the fallback message |

## Hybrid retrieval + rerank

`rag_hybrid_rerank.py` changes only the retriever compared with `heading_aware` (same data, chunking, prompt, generator,
and `top_k=4`), so the score difference comes from the retriever. Dense and BM25 each return 20 candidates, Reciprocal
Rank Fusion merges them, and `gpt-4o-mini` scores the top 20 from 0 to 3 for how directly they answer the question.

```bash
python rag_hybrid_rerank.py --dry-run     # answers and source_hit only, no judge calls
python rag_hybrid_rerank.py               # RAGAS on the shared test set -> hybrid_rerank
python rag_hybrid_rerank.py --no-rerank   # hybrid without the reranker -> hybrid
python rag_hybrid_rerank.py --official    # RAGAS on the 12 Canvas questions -> hybrid_rerank_official
python rag_hybrid_rerank.py --prompt v2   # revised generation prompt -> hybrid_rerank_prompt_v2
```

| Run | Test set | `context_precision` | `context_recall` | `faithfulness` | `answer_relevancy` | `factual_correctness` |
|---|---|---|---|---|---|---|
| `heading_aware` | shared | 0.60 | 0.76 | 0.98 | 0.76 | 0.55 |
| `hybrid_rerank` | shared | 0.73 | 0.77 | 0.98 | 0.76 | 0.55 |
| `hybrid_rerank_prompt_v2` | shared | 0.85 | 0.77 | 0.97 | 0.67 | 0.50 |
| `hybrid_rerank_prompt_v3` | shared | 0.76 | 0.79 | 0.95 | 0.66 | 0.55 |
| `hybrid_rerank_official` | official | 0.94 | 0.93 | 0.97 | 0.80 | 0.64 |
| `hybrid_rerank_prompt_v2_official` | official | 0.90 | 0.93 | 0.92 | 0.74 | 0.51 |
| `hybrid_rerank_prompt_v3_official` | official | 0.85 | 0.93 | 0.90 | 0.77 | 0.57 |

- The `_official` runs use a different test set, so compare them only with each other.
- `v1` stays the default prompt because `v2` didn't score higher. Much of that gap looks like judge noise: nearly
  identical deadline lists scored 0.92 and 0.00. RAGAS also marks "the context doesn't specify" lines as noncommittal.
  With this few questions, average two or more runs before choosing between prompts.
- `v3` is `v1` plus three targeted rules: passages that use different words than the question still count, actions
  such as "Schedule an Appointment" are passed on, and partial answers are given. `v1` refused the official question
  about an advising appointment on the live app even though the admissions-counselor chunk ranked first. `v3` answers
  it (`factual_correctness` 0.57 to 0.83 on that question) and still refuses both out-of-scope questions, but its
  average isn't higher: 0.57 vs. 0.64 on the official set, 0.55 for both on the shared set, and `faithfulness` is
  lower on both. These are single runs. The app uses `v3` because that question is on the official list; average
  several runs of `v1` and `v3` before treating either as better.

## Team workflow

1. Create your own branch from `main`:
   ```bash
   git checkout main && git pull
   git checkout -b <your-name>/<approach>      # e.g. maria/hybrid-search
   ```
2. Change the pipeline. This can be the chunking, retriever, reranker, prompt, or model, among others. Work in a copy
   of the notebook (for example `rag_<approach>.ipynb`) or in your own module.
3. Evaluate the change with `rag_eval`, and give the run a **unique run name**:
   ```python
   import rag_eval
   testset = rag_eval.load_testset()
   runs = rag_eval.collect(lambda q: ask(q, verbose=False), testset)  # ask() returns answer, contexts, sources
   scores = rag_eval.score(runs, fallback=FALLBACK)
   rag_eval.save_results(scores, "hybrid_search", config={"retriever": "bm25 + dense", "top_k": 4})
   rag_eval.leaderboard()
   ```
4. Commit your code together with `eval/results/<run_name>.json` and `.csv`, push the branch, and open a pull request.
   After merging, `rag_eval.leaderboard()` shows every approach side by side.

## Status

- [x] Scraper for the program site
- [x] Baseline RAG: recursive chunking, Chroma, `gpt-4o-mini`, source citations, fallback for unanswerable questions
- [x] Improvement 1: heading-aware chunks with breadcrumbs and outline chunks
- [x] RAGAS evaluation with a shared test set
- [x] Retrieval improvements: hybrid search, reranking
- [x] Responsible AI: PII redaction, groundedness check (shown as a warning, not blocking)
- [x] Streamlit UI
- [x] Deployment: Streamlit Community Cloud, https://msads-rag-chatbot.streamlit.app/
- [ ] Documentation and presentation
