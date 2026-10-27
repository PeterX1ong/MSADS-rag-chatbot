"""
Shared RAGAS evaluation for the MS in Applied Data Science RAG chatbot.

Every RAG variant (baseline, heading-aware chunks, reranking, ...) is scored the same way so results are comparable
across branches:

    import rag_eval
    testset = rag_eval.load_testset()
    runs = rag_eval.collect(lambda q: ask(q, verbose=False), testset)   # ask() must return answer/contexts/sources
    scores = rag_eval.score(runs, fallback=FALLBACK)
    rag_eval.save_results(scores, run_name="baseline", config={...})
    rag_eval.leaderboard()                                              # every saved run, side by side

Metrics (in-scope questions):
    context_precision    retrieval: are the chunks that support the reference ranked near the top?   (RAGAS, LLM judge)
    context_recall       retrieval: do the retrieved chunks contain everything in the reference?      (RAGAS, LLM judge)
    faithfulness         generation: is every claim in the answer supported by the retrieved chunks?  (RAGAS, LLM judge)
    answer_relevancy     generation: does the answer address the question that was asked?            (RAGAS, embeddings)
    factual_correctness  end-to-end: claim-level F1 of the answer against the reference answer        (RAGAS, LLM judge)
    source_hit           retrieval: at least one retrieved chunk comes from a reference page          (no LLM)
    answered             the pipeline answered instead of returning the fallback message              (no LLM)
Out-of-scope questions are only checked for `refused` (the fallback message was returned).
"""
import json
import math
import os
import re
from datetime import datetime
from pathlib import Path

import pandas as pd

os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")  # opt out of RAGAS usage analytics

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import EvaluationDataset, RunConfig, evaluate
from ragas.cost import get_token_usage_for_openai
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    FactualCorrectness,
    Faithfulness,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
    ResponseRelevancy,
)

EVAL_DIR = Path(__file__).parent / "eval"
TESTSET_PATH = EVAL_DIR / "testset.json"
RESULTS_DIR = EVAL_DIR / "results"

# Keep these fixed across branches -- changing the judge changes every score.
JUDGE_MODEL = "gpt-4o"
JUDGE_EMBED_MODEL = "text-embedding-3-small"
JUDGE_PRICE_PER_1M = {"input": 2.50, "output": 10.00}  # USD, gpt-4o; only used for the cost estimate

METRICS =["context_precision", "context_recall", "faithfulness", "answer_relevancy", "factual_correctness",
           "source_hit", "answered"]


def load_testset(path=TESTSET_PATH) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)["questions"]


def collect(ask_fn, testset) -> list[dict]:
    """Run every test question through a pipeline. `ask_fn(question)` returns a dict with answer/contexts/sources."""
    runs = []
    for item in testset:
        out = ask_fn(item["question"])
        runs.append({**item, "answer": out["answer"], "contexts": out["contexts"], "sources": out["sources"]})
    return runs


def strip_sources(answer: str) -> str:
    """Drop the trailing 'Sources: <urls>' line so the judge scores the answer itself, not the citation list."""
    return re.split(r"\n\s*\**Sources?\**:", answer)[0].strip()


def _ragas_scores(runs, judge_model, embed_model):
    dataset = EvaluationDataset.from_list([
        {"user_input": r["question"], "retrieved_contexts": r["contexts"],
         "response": strip_sources(r["answer"]), "reference": r["reference"]}
        for r in runs
    ])
    metrics = {  # our column name -> RAGAS metric
        "context_precision": LLMContextPrecisionWithReference(),
        "context_recall": LLMContextRecall(),
        "faithfulness": Faithfulness(),
        "answer_relevancy": ResponseRelevancy(),
        "factual_correctness": FactualCorrectness(mode="f1"),
    }
    result = evaluate(
        dataset,
        metrics=list(metrics.values()),
        llm=LangchainLLMWrapper(ChatOpenAI(model=judge_model, temperature=0)),
        embeddings=LangchainEmbeddingsWrapper(OpenAIEmbeddings(model=embed_model)),
        run_config=RunConfig(max_workers=8, timeout=180),
        token_usage_parser=get_token_usage_for_openai,
    )
    try:
        cost = result.total_cost(cost_per_input_token=JUDGE_PRICE_PER_1M["input"] / 1e6,
                                 cost_per_output_token=JUDGE_PRICE_PER_1M["output"] / 1e6)
        print(f"Judge cost for this run: ~${cost:.2f}")
    except Exception as e:  # the estimate is informational; never lose the scores over it
        print(f"Judge cost unavailable ({e})")
    columns = {metric.name: label for label, metric in metrics.items()}  # read names after evaluate() runs
    return result.to_pandas().rename(columns=columns)[list(metrics)]


def score(runs, fallback: str, judge_model=JUDGE_MODEL, embed_model=JUDGE_EMBED_MODEL) -> pd.DataFrame:
    """Per-question scores: RAGAS metrics + source hit + answered/refused checks."""
    df = pd.DataFrame(runs)
    df["answered"] = ~df["answer"].str.contains(fallback, regex=False)
    df["refused"] = ~df["answered"]
    df["source_hit"] = [bool(set(r["sources"]) & set(r["reference_sources"])) for r in runs]

    in_scope = df["in_scope"]
    ragas_df = _ragas_scores([r for r in runs if r["in_scope"]], judge_model, embed_model)
    ragas_df.index = df.index[in_scope]
    df = df.join(ragas_df)
    df.loc[~in_scope, ["answered", "source_hit"]] = None  # not meaningful for out-of-scope questions
    return df[["id", "question", "in_scope", *METRICS, "refused", "answer", "sources"]]


def summarize(scores: pd.DataFrame) -> dict:
    """Mean of each metric over in-scope questions, plus the refusal rate on out-of-scope questions."""
    in_scope = scores[scores["in_scope"]]
    summary = {m: float(in_scope[m].astype(float).mean()) for m in METRICS}
    out_scope = scores[~scores["in_scope"]]
    summary["out_of_scope_refusal"] = float(out_scope["refused"].mean()) if len(out_scope) else math.nan
    return summary


def save_results(scores: pd.DataFrame, run_name: str, config: dict | None = None) -> dict:
    """Write eval/results/<run_name>.csv (per question) and <run_name>.json (summary + config)."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    scores.to_csv(RESULTS_DIR / f"{run_name}.csv", index=False)
    record = {
        "run_name": run_name,
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        "judge_model": JUDGE_MODEL,
        "n_questions": int(len(scores)),
        "config": config or {},
        "summary": summarize(scores),
    }
    with open(RESULTS_DIR / f"{run_name}.json", "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
    print(f"Saved {RESULTS_DIR / run_name}.csv/.json")
    return record


def leaderboard(results_dir=None) -> pd.DataFrame:
    """One row per saved run (every <run_name>.json in eval/results/), metrics as columns."""
    rows = []
    for path in sorted(Path(results_dir or RESULTS_DIR).glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        rows.append({"run_name": record["run_name"], **record["summary"], "evaluated_at": record["evaluated_at"]})
    return pd.DataFrame(rows).set_index("run_name") if rows else pd.DataFrame()
