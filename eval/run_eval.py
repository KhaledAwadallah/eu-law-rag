"""Evaluation harness: score the RAG pipeline on eval/questions.jsonl.

Metrics:
    retrieval hit-rate   did the expected paper appear in the top-k? (no LLM needed)
    false refusals       answerable questions the system wrongly refused
    refusal accuracy     trap questions the system correctly refused
    citation rate        answered questions containing [n] citations
    faithfulness         LLM-as-judge: are the answer's claims supported by the excerpts?

Usage (from the project root, Ollama running unless --no-llm):
    python eval/run_eval.py                     # full evaluation, results named by k
    python eval/run_eval.py --no-llm            # retrieval metrics only, fast
    python eval/run_eval.py --k 10 --name k10   # experiment: different retrieval depth
"""

import argparse
import json
import pathlib
import re
import statistics
import time

from askarxiv import config
from askarxiv.generate import REFUSAL, answer, call_llm, format_context

QUESTIONS_FILE = pathlib.Path(__file__).parent / "questions.jsonl"
RESULTS_DIR = pathlib.Path(__file__).parent / "results"

JUDGE_TEMPLATE = """You are grading a research assistant's answer against its sources.

Excerpts the assistant was given:
{context}

The assistant's answer:
{answer}

Is every factual claim in the answer supported by the excerpts?
Reply with exactly one word:
UNSUPPORTED if the main claims are not backed by the excerpts.
PARTIAL if some claims are backed and some are not.
SUPPORTED if every claim is backed by the excerpts."""


def load_questions(path: pathlib.Path = QUESTIONS_FILE) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def parse_verdict(reply: str) -> tuple[str, float]:
    """Map the judge's reply to (label, score). Order matters:
    'SUPPORTED' is a substring of 'UNSUPPORTED', so test the longer word first."""
    text = reply.strip().upper()
    for label, score in (("UNSUPPORTED", 0.0), ("PARTIAL", 0.5), ("SUPPORTED", 1.0)):
        if label in text:
            return label, score
    return "UNPARSEABLE", 0.0


def judge_faithfulness(context: str, answer_text: str) -> tuple[str, float]:
    reply = call_llm(JUDGE_TEMPLATE.format(context=context, answer=answer_text))
    return parse_verdict(reply)


def evaluate(k: int, use_llm: bool) -> tuple[list[dict], dict]:
    from askarxiv.retrieve import retrieve

    rows = []
    for q in load_questions():
        hits = retrieve(q["question"], k=k)
        retrieved_ids = {h["paper_id"] for h in hits}
        row = {"id": q["id"], "answerable": q["answerable"]}
        if q["answerable"]:
            row["retrieval_hit"] = bool(set(q["source_ids"]) & retrieved_ids)

        if use_llm:
            result = answer(q["question"], k=k)
            text = result["answer"]
            row["refused"] = REFUSAL in text
            row["cited"] = bool(re.search(r"\[\d+\]", text))
            if q["answerable"] and not row["refused"]:
                label, score = judge_faithfulness(
                    format_context(result["sources"]), text)
                row["verdict"], row["faithfulness"] = label, score
            row["answer"] = text

        rows.append(row)
        status = "hit " if row.get("retrieval_hit") else ("miss" if q["answerable"] else "trap")
        print(f"{q['id']} [{status}] refused={row.get('refused', '-')} "
              f"verdict={row.get('verdict', '-')}")
    return rows, summarize(rows, k, use_llm)


def summarize(rows: list[dict], k: int, use_llm: bool) -> dict:
    answerable = [r for r in rows if r["answerable"]]
    traps = [r for r in rows if not r["answerable"]]

    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M"),
        "n_questions": len(rows),
        "config": {
            "k": k,
            "chunk_size": config.CHUNK_SIZE,
            "chunk_overlap": config.CHUNK_OVERLAP,
            "max_chunks_per_paper": config.MAX_CHUNKS_PER_PAPER,
            "embedding_model": config.EMBEDDING_MODEL,
            "llm_model": config.LLM_MODEL if use_llm else None,
        },
        "retrieval_hit_rate": statistics.mean(
            r["retrieval_hit"] for r in answerable),
    }
    if use_llm:
        answered = [r for r in answerable if not r["refused"]]
        judged = [r for r in answered if "faithfulness" in r]
        summary.update({
            "false_refusal_rate": statistics.mean(r["refused"] for r in answerable),
            "refusal_accuracy": statistics.mean(r["refused"] for r in traps),
            "citation_rate": statistics.mean(r["cited"] for r in answered) if answered else None,
            "faithfulness": statistics.mean(r["faithfulness"] for r in judged) if judged else None,
        })
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the AskArxiv pipeline.")
    parser.add_argument("--k", type=int, default=config.TOP_K,
                        help="retrieval depth (default: config.TOP_K)")
    parser.add_argument("--no-llm", action="store_true",
                        help="retrieval metrics only; no generation or judging")
    parser.add_argument("--name", default=None,
                        help="results file name (default: k<k> or k<k>-nollm)")
    args = parser.parse_args()

    rows, summary = evaluate(k=args.k, use_llm=not args.no_llm)

    name = args.name or (f"k{args.k}" + ("-nollm" if args.no_llm else ""))
    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"{name}.json"
    out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2),
                   encoding="utf-8")

    print("\n=== summary ===")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"{key:>22}: {value:.2f}")
        elif key != "config":
            print(f"{key:>22}: {value}")
    print(f"{'config':>22}: {summary['config']}")
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
