"""Scores the pipeline on eval/questions.jsonl.

Retrieval is scored by provision hit-rate: does the top-k contain the provision
that answers the question? With an LLM, the run also scores false refusals,
refusal accuracy on trap questions, citation rate, misattribution and
faithfulness (judged by the same LLM, so treat it as an upper bound).

Usage:
    python -m eval.run_eval --no-llm   # retrieval only, no LLM needed
    python -m eval.run_eval            # full run
    python -m eval.run_eval --k 10     # another retrieval depth
    python -m eval.run_eval --plain    # plain top-k, without the ranking rules
"""

import argparse
import json
import pathlib
import re
import statistics
import sys
import time

from eulaw import config
from eulaw import generate
from eulaw import retrieve

QUESTIONS_FILE = pathlib.Path(__file__).parent / "questions.jsonl"
RESULTS_DIR = pathlib.Path(__file__).parent / "results"

# Models cite as [1], 【3】 or 【1†L1-L3】.
_CITATION = re.compile(r"[\[【]\d+[^\[\]【】]{0,30}[\]】]")

# Laws and rulings that are not in the corpus. An answer that names one is
# presenting corpus text as something it is not.
_FOREIGN_LAWS = (
    "Cyber Resilience Act",
    "Digital Services Act",
    "Digital Markets Act",
    "Data Governance Act",
    "Data Act",
    "NIS2",
    "ePrivacy",
    "Product Liability Directive",
    "Consumer Privacy Act",
    "Executive Order",
    "Schrems",
)

_JUDGE_TEMPLATE = """\
You are grading a research assistant's answer against its sources.

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
    """Reads the question set, one JSON object per line."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def parse_verdict(reply: str) -> tuple[str, float]:
    """Maps the judge's reply to a label and a score in [0, 1]."""
    text = reply.strip().upper()
    # UNSUPPORTED first: it contains SUPPORTED.
    for label, score in (("UNSUPPORTED", 0.0), ("PARTIAL", 0.5),
                         ("SUPPORTED", 1.0)):
        if label in text:
            return label, score
    return "UNPARSEABLE", 0.0


def misattributed(text: str) -> list[str]:
    """Lists the laws outside the corpus that an answer names.

    A plain string match, so it cannot share the generator's blind spots.
    """
    return [law for law in _FOREIGN_LAWS
            if re.search(rf"\b{re.escape(law)}\b", text, re.IGNORECASE)]


def provision_hit(question: dict, hits: list[dict]) -> bool:
    """Whether the hits include a gold provision of the question."""
    return any(hit["doc_id"] in question["source_ids"]
               and hit["label"] in question["source_labels"]
               for hit in hits)


def score_question(question: dict, k: int, use_llm: bool,
                   plain: bool) -> dict:
    """Runs one question through the pipeline and scores it.

    Args:
        question: One entry of the question set.
        k: Retrieval depth.
        use_llm: Whether to generate and judge an answer.
        plain: Whether to use plain top-k instead of the ranking rules.

    Returns:
        The scores, plus the retrieved sources and the answer, so the run can
        be re-analysed without calling the LLM again.
    """
    rules = {"max_per_provision": k, "min_primary": 0.0} if plain else {}
    hits = retrieve.retrieve(question["question"], k=k, **rules)
    row = {"id": question["id"], "answerable": question["answerable"]}
    if question["answerable"]:
        row["provision_hit"] = provision_hit(question, hits)
    row["sources"] = [{"doc_id": h["doc_id"], "label": h["label"],
                       "score": h["score"]} for h in hits]
    if not use_llm:
        return row

    text = generate.call_llm(generate.build_prompt(question["question"], hits))
    row["refused"] = generate.REFUSAL in text
    row["cited"] = bool(_CITATION.search(text))
    row["misattributed"] = misattributed(text)
    if question["answerable"] and not row["refused"]:
        reply = generate.call_llm(_JUDGE_TEMPLATE.format(
            context=generate.format_context(hits), answer=text))
        row["verdict"], row["faithfulness"] = parse_verdict(reply)
    row["answer"] = text
    return row


def _mean(values: list[bool | float]) -> float | None:
    return statistics.mean(values) if values else None


def summarize(rows: list[dict], k: int, use_llm: bool, plain: bool) -> dict:
    """Aggregates per-question rows into the headline metrics."""
    answerable = [r for r in rows if r["answerable"]]
    traps = [r for r in rows if not r["answerable"]]
    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M"),
        "n_questions": len(rows),
        "n_answerable": len(answerable),
        "n_traps": len(traps),
        "config": {
            "k": k,
            "ranking_rules": not plain,
            "max_chunks_per_provision": config.MAX_CHUNKS_PER_PROVISION,
            "min_primary": config.MIN_PRIMARY,
            "embedding_model": config.EMBEDDING_MODEL,
            "llm_model": config.LLM_MODEL if use_llm else None,
        },
        "provision_hit_rate": _mean([r["provision_hit"] for r in answerable]),
    }
    if use_llm:
        answered = [r for r in answerable if not r["refused"]]
        summary.update({
            "false_refusal_rate": _mean([r["refused"] for r in answerable]),
            "refusal_accuracy": _mean([r["refused"] for r in traps]),
            "citation_rate": _mean([r["cited"] for r in answered]),
            "misattribution_rate": _mean(
                [bool(r["misattributed"]) for r in rows]),
            "faithfulness": _mean(
                [r["faithfulness"] for r in answered if "faithfulness" in r]),
        })
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the RAG pipeline.")
    parser.add_argument("--k", type=int, default=config.TOP_K,
                        help="retrieval depth")
    parser.add_argument("--no-llm", action="store_true",
                        help="retrieval metrics only; no generation")
    parser.add_argument("--plain", action="store_true",
                        help="plain top-k, without the two ranking rules")
    args = parser.parse_args()
    # Legal text has characters a Windows console cannot always encode.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    use_llm = not args.no_llm

    rows = []
    for question in load_questions():
        row = score_question(question, args.k, use_llm, args.plain)
        rows.append(row)
        print(f"{row['id']} provision_hit={row.get('provision_hit', '-')} "
              f"refused={row.get('refused', '-')} "
              f"verdict={row.get('verdict', '-')}")
    summary = summarize(rows, args.k, use_llm, args.plain)

    name = f"k{args.k}"
    if args.plain:
        name += "-plain"
    if not use_llm:
        name += "-nollm"
    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"{name}.json"
    out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2),
                   encoding="utf-8")

    print()
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"{key:>20}: {value:.2f}")
        else:
            print(f"{key:>20}: {value}")
    print(f"\nsaved to {out}")


if __name__ == "__main__":
    main()
