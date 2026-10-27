# MS in Applied Data Science RAG Chatbot

ADSP 32027 Generative AI Principles, Course Project 1.

This is a retrieval-augmented generation (RAG) chatbot that answers questions about the
[UChicago MS in Applied Data Science](https://datascience.uchicago.edu/education/masters-programs/ms-in-applied-data-science/)
program using only the program's website. Answers cite the source pages. When the site doesn't contain the answer,
the chatbot says so instead of guessing.

`main` holds the **baseline** pipeline and a shared **RAGAS evaluation**. Each teammate builds an alternative approach
on their own branch and scores it with the same evaluation, so we can compare approaches and pick the best.

## Repository layout

| Path | What it is |
|---|---|
| `scraper.py` | Crawls the program site (main page, sublinks, faculty profiles, MS-ADS news) into `data/` |
| `data/pages.jsonl` | Scraped pages, one JSON record per page (`url`, `title`, `text`, ...). Committed so everyone uses the same data |
| `data/raw_pages/*.txt` | The same pages as readable text files |
| `rag_baseline.ipynb` | End-to-end pipeline: scrape, chunk, embed into Chroma, answer with GPT, evaluate |
| `rag_eval.py` | Shared evaluation module (RAGAS metrics, saving results, leaderboard) |
| `eval/testset.json` | Shared test set: 15 in-scope questions with hand-written reference answers, plus 2 out-of-scope questions |
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
   - `RUN_SCRAPER = True` re-crawls the site, which takes about 2 minutes. Set it to `False` to use the committed
     `data/pages.jsonl`. When comparing approaches, use `False` so everyone works from the same data.
   - `chroma_db/` is rebuilt automatically and isn't committed.

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
- [ ] Retrieval improvements: hybrid search, reranking
- [ ] Responsible AI: PII redaction, groundedness guardrails
- [ ] Streamlit UI
- [ ] Documentation and presentation
