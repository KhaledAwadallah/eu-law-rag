"""Score the pipeline on eval/questions.jsonl.

Metrics: retrieval and provision hit-rate, false refusals, refusal accuracy on
traps, citation rate, misattribution, and LLM-judged faithfulness.

    python eval/run_eval.py                     # full run
    python eval/run_eval.py --no-llm            # retrieval metrics only, fast
    python eval/run_eval.py --k 10 --name k10   # try another retrieval depth
"""

import argparse
import json
import pathlib
import re
import statistics
import sys
import time

from eulaw import config
from eulaw.generate import REFUSAL, answer, call_llm, format_context

QUESTIONS_FILE = pathlib.Path(__file__).parent / "questions.jsonl"
RESULTS_DIR = pathlib.Path(__file__).parent / "results"

# Models cite as [1], 【3】 or 【1†L1-L3】. Requiring a digit immediately before
# the closing bracket under-counted the last form, so allow an annotation.
CITATION_PATTERN = r"[\[【]\d+[^\[\]【】]{0,30}[\]】]"

# EU instruments absent from the corpus. An answer asserting what one of these
# requires is relabelling corpus text as a law it never saw.
FOREIGN_INSTRUMENTS = (
    "Cyber Resilience Act",
    "Digital Services Act",
    "Digital Markets Act",
    "Data Governance Act",
    "NIS2",
    "ePrivacy",
)

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
    """Order matters: 'SUPPORTED' is a substring of 'UNSUPPORTED'."""
    text = reply.strip().upper()
    for label, score in (("UNSUPPORTED", 0.0), ("PARTIAL", 0.5), ("SUPPORTED", 1.0)):
        if label in text:
            return label, score
    return "UNPARSEABLE", 0.0


def misattributed(text: str) -> list[str]:
    """A plain string check, so it cannot inherit the generator's blind spots."""
    return [name for name in FOREIGN_INSTRUMENTS if name.lower() in text.lower()]


def judge_faithfulness(context: str, answer_text: str) -> tuple[str, float]:
    reply = call_llm(JUDGE_TEMPLATE.format(context=context, answer=answer_text))
    return parse_verdict(reply)


def evaluate(k: int, use_llm: bool) -> tuple[list[dict], dict]:
    from eulaw.retrieve import retrieve

    rows = []
    for q in load_questions():
        hits = retrieve(q["question"], k=k)
        row = {"id": q["id"], "answerable": q["answerable"]}

        if q["answerable"]:
            row["retrieval_hit"] = bool(set(q["source_ids"]) &
                                        {h["doc_id"] for h in hits})
            # Document-level is trivial with two regulations; score the provision.
            if q.get("source_labels"):
                row["provision_hit"] = bool(set(q["source_labels"]) &
                                            {h["label"] for h in hits})

        # Keep the excerpts, not just the score: re-analysis then costs no
        # further LLM calls.
        row["sources"] = [{"doc_id": h["doc_id"], "label": h["label"],
                           "score": h["score"]} for h in hits]

        if use_llm:
            result = answer(q["question"], k=k)
            text = result["answer"]
            row["refused"] = REFUSAL in text
            row["cited"] = bool(re.search(CITATION_PATTERN, text))
            row["misattributed"] = misattributed(text)
            if q["answerable"] and not row["refused"]:
                label, score = judge_faithfulness(
                    format_context(result["sources"]), text)
                row["verdict"], row["faithfulness"] = label, score
            row["answer"] = text

        rows.append(row)
        status = "hit " if row.get("retrieval_hit") else ("miss" if q["answerable"] else "trap")
        print(f"{q['id']} [{status}] provision={row.get('provision_hit', '-')} "
              f"refused={row.get('refused', '-')} verdict={row.get('verdict', '-')} "
              f"misattr={row.get('misattributed', '-')}")
    return rows, summarize(rows, k, use_llm)


def _mean(values: list) -> float | None:
    return statistics.mean(values) if values else None


def summarize(rows: list[dict], k: int, use_llm: bool) -> dict:
    answerable = [r for r in rows if r["answerable"]]
    traps = [r for r in rows if not r["answerable"]]

    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M"),
        "n_questions": len(rows),
        "config": {
            "k": k,
            "documents": [d["celex"] for d in config.DOCUMENTS],
            "chunk_size": config.CHUNK_SIZE,
            "chunk_overlap": config.CHUNK_OVERLAP,
            "max_chunks_per_provision": config.MAX_CHUNKS_PER_PROVISION,
            "min_primary": config.MIN_PRIMARY,
            "embedding_model": config.EMBEDDING_MODEL,
            "llm_model": config.LLM_MODEL if use_llm else None,
        },
        "retrieval_hit_rate": _mean([r["retrieval_hit"] for r in answerable]),
        "provision_hit_rate": _mean(
            [r["provision_hit"] for r in answerable if "provision_hit" in r]),
    }
    if use_llm:
        answered = [r for r in answerable if not r["refused"]]
        judged = [r for r in answered if "faithfulness" in r]
        summary.update({
            "false_refusal_rate": _mean([r["refused"] for r in answerable]),
            "refusal_accuracy": _mean([r["refused"] for r in traps]),
            "citation_rate": _mean([r["cited"] for r in answered]),
            "misattribution_rate": _mean(
                [bool(r["misattributed"]) for r in rows]),
            "faithfulness": _mean([r["faithfulness"] for r in judged]),
        })
    return summary


def main() -> None:
    # Legal text has characters the Windows console cannot encode by default.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Evaluate the RAG pipeline.")
    parser.add_argument("--k", type=int, default=config.TOP_K,
                        help="retrieval depth (default: config.TOP_K)")
    parser.add_argument("--no-llm", action="store_true",
                        help="retrieval metrics only; no generation or judging")
    parser.add_argument("--name", default=None,
                        help="results file name (default: k<k>)")
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
