"""
Unified Evaluation for HAUI Agent — BLEU, ROUGE & Recall@K.

Reads question-answer pairs from data/qa.parquet and evaluates:
  - BLEU-1 to BLEU-4  (generation quality — n-gram precision)
  - ROUGE-1, ROUGE-2, ROUGE-L  (generation quality — overlap)
  - Recall@K  (retrieval quality — chunk hit rate)

Usage:
    # Run all metrics (BLEU + ROUGE need LLM API; Recall@K needs only embedding API)
    python scripts/evaluate.py --mode all --limit 10

    # Run only retrieval evaluation (no LLM credits needed)
    python scripts/evaluate.py --mode recall --store curriculum --k 1 3 5 10

    # Run only generation evaluation (BLEU + ROUGE)
    python scripts/evaluate.py --mode generation --limit 5

    # Run individual metrics
    python scripts/evaluate.py --mode bleu --limit 5
    python scripts/evaluate.py --mode rouge --limit 5
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime

# ── Fix Windows console encoding for Vietnamese text ──────────────────────────
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr.encoding != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

# ── Ensure project root is on sys.path ────────────────────────────────────────
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from dotenv import load_dotenv

load_dotenv(os.path.join(ROOT_DIR, ".env"))

logger = logging.getLogger(__name__)

SMOOTHING = SmoothingFunction().method1


# ═══════════════════════════════════════════════════════════════════════════════
# Tokeniser
# ═══════════════════════════════════════════════════════════════════════════════

def tokenize_vi(text: str) -> list[str]:
    """Tokenize Vietnamese text by whitespace splitting and lowercasing."""
    return text.lower().strip().split()


class ViTokenizer:
    """Whitespace tokenizer compatible with rouge_score library."""

    def tokenize(self, text: str) -> list[str]:
        return tokenize_vi(text)


# ═══════════════════════════════════════════════════════════════════════════════
# Data extraction helpers
# ═══════════════════════════════════════════════════════════════════════════════

def extract_references(generation_gt) -> list[str]:
    """Extract non-empty reference answer strings from the generation_gt field."""
    refs: list[str] = []
    if isinstance(generation_gt, np.ndarray):
        items = generation_gt.tolist()
    elif isinstance(generation_gt, list):
        items = generation_gt
    elif isinstance(generation_gt, str):
        items = [generation_gt]
    else:
        return refs
    for item in items:
        if isinstance(item, str) and item.strip():
            refs.append(item.strip())
    return refs


def extract_gt_chunk_ids(retrieval_gt) -> set[str]:
    """Extract ground-truth chunk IDs from the retrieval_gt field."""
    ids: set[str] = set()
    if isinstance(retrieval_gt, np.ndarray):
        for item in retrieval_gt.flat:
            if isinstance(item, np.ndarray):
                for sub in item.flat:
                    if isinstance(sub, str) and sub.strip():
                        ids.add(sub.strip())
            elif isinstance(item, str) and item.strip():
                ids.add(item.strip())
    elif isinstance(retrieval_gt, list):
        for item in retrieval_gt:
            if isinstance(item, list):
                for sub in item:
                    if isinstance(sub, str) and sub.strip():
                        ids.add(sub.strip())
            elif isinstance(item, str) and item.strip():
                ids.add(item.strip())
    return ids


# ═══════════════════════════════════════════════════════════════════════════════
# BLEU scoring
# ═══════════════════════════════════════════════════════════════════════════════

def compute_bleu(references: list[str], candidate: str) -> dict[str, float]:
    """Compute BLEU-1 through BLEU-4 for a candidate vs references."""
    ref_tokens = [tokenize_vi(ref) for ref in references if ref.strip()]
    cand_tokens = tokenize_vi(candidate)

    if not ref_tokens or not cand_tokens:
        return {"bleu_1": 0.0, "bleu_2": 0.0, "bleu_3": 0.0, "bleu_4": 0.0}

    weights = {
        "bleu_1": (1.0, 0, 0, 0),
        "bleu_2": (0.5, 0.5, 0, 0),
        "bleu_3": (1 / 3, 1 / 3, 1 / 3, 0),
        "bleu_4": (0.25, 0.25, 0.25, 0.25),
    }

    scores = {}
    for name, w in weights.items():
        scores[name] = round(sentence_bleu(
            ref_tokens, cand_tokens, weights=w, smoothing_function=SMOOTHING,
        ), 4)
    return scores


# ═══════════════════════════════════════════════════════════════════════════════
# ROUGE scoring
# ═══════════════════════════════════════════════════════════════════════════════

def compute_rouge(references: list[str], candidate: str) -> dict[str, dict[str, float]]:
    """Compute ROUGE-1, ROUGE-2, ROUGE-L (best across references)."""
    if not references or not candidate.strip():
        empty = {"precision": 0.0, "recall": 0.0, "fmeasure": 0.0}
        return {"rouge1": empty, "rouge2": empty, "rougeL": empty}

    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], tokenizer=ViTokenizer(),
    )

    best: dict[str, dict[str, float]] = {}
    for ref in references:
        if not ref.strip():
            continue
        scores = scorer.score(ref, candidate)
        for name, obj in scores.items():
            cur = {
                "precision": round(obj.precision, 4),
                "recall": round(obj.recall, 4),
                "fmeasure": round(obj.fmeasure, 4),
            }
            if name not in best or cur["fmeasure"] > best[name]["fmeasure"]:
                best[name] = cur

    if not best:
        empty = {"precision": 0.0, "recall": 0.0, "fmeasure": 0.0}
        return {"rouge1": empty, "rouge2": empty, "rougeL": empty}
    return best


# ═══════════════════════════════════════════════════════════════════════════════
# Recall@K scoring
# ═══════════════════════════════════════════════════════════════════════════════

def compute_recall_at_k(gt_ids: set[str], retrieved_ids: list[str], k: int) -> float:
    """Compute Recall@K."""
    if not gt_ids:
        return 0.0
    top_k = set(str(cid) for cid in retrieved_ids[:k])
    return len(top_k & set(str(g) for g in gt_ids)) / len(gt_ids)


def route_query(query: str) -> str:
    """Simple keyword-based routing for retrieval domain."""
    q = query.lower()
    kws = [
        "quy chế", "quy định", "chính sách", "phòng ban",
        "giới thiệu", "chiến lược", "tổ chức", "giảng viên",
        "sự kiện", "thông báo", "chất lượng", "đội ngũ", "cơ cấu",
    ]
    return "regulations" if any(k in q for k in kws) else "curriculum"


# ═══════════════════════════════════════════════════════════════════════════════
# Generation evaluation (BLEU + ROUGE)
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_generation(
    df: pd.DataFrame,
    run_bleu: bool = True,
    run_rouge: bool = True,
) -> tuple[list[dict], dict]:
    """Evaluate generation quality using BLEU and/or ROUGE.

    Returns (per_question_results, aggregate_scores).
    """
    from src.agent.haui_agent import HAUIAgent

    print("\n  Initialising HAUIAgent...")
    agent = HAUIAgent()

    results: list[dict] = []
    bleu_accum = {"bleu_1": [], "bleu_2": [], "bleu_3": [], "bleu_4": []}
    rouge_accum = {
        "rouge1_p": [], "rouge1_r": [], "rouge1_f": [],
        "rouge2_p": [], "rouge2_r": [], "rouge2_f": [],
        "rougeL_p": [], "rougeL_r": [], "rougeL_f": [],
    }

    total = len(df)

    for idx, row in df.iterrows():
        qid = row.get("qid", str(idx))
        question = row["query"]
        refs = extract_references(row["generation_gt"])

        if not refs:
            print(f"  [{idx + 1}/{total}] SKIP (no ground truth)")
            continue

        print(f"\n  [{idx + 1}/{total}] Q: {question[:80]}...")

        agent.reset()
        t0 = time.time()
        try:
            resp = agent.chat(question)
            candidate = resp.answer
            intent = resp.intent
        except Exception as exc:
            print(f"    ❌ ERROR: {exc}")
            candidate, intent = "", "error"
        elapsed = time.time() - t0

        entry: dict = {
            "qid": qid,
            "question": question,
            "reference": refs[0][:200] + ("..." if len(refs[0]) > 200 else ""),
            "candidate": candidate[:200] + ("..." if len(candidate) > 200 else ""),
            "intent": intent,
            "elapsed_seconds": round(elapsed, 2),
        }

        # ── BLEU ──
        if run_bleu:
            bleu = compute_bleu(refs, candidate)
            entry["bleu"] = bleu
            for k, v in bleu.items():
                bleu_accum[k].append(v)
            print(f"    BLEU  1:{bleu['bleu_1']:.4f}  2:{bleu['bleu_2']:.4f}  "
                  f"3:{bleu['bleu_3']:.4f}  4:{bleu['bleu_4']:.4f}")

        # ── ROUGE ──
        if run_rouge:
            rouge = compute_rouge(refs, candidate)
            entry["rouge"] = rouge
            for metric in ["rouge1", "rouge2", "rougeL"]:
                rouge_accum[f"{metric}_p"].append(rouge[metric]["precision"])
                rouge_accum[f"{metric}_r"].append(rouge[metric]["recall"])
                rouge_accum[f"{metric}_f"].append(rouge[metric]["fmeasure"])
            r1, r2, rL = rouge["rouge1"], rouge["rouge2"], rouge["rougeL"]
            print(f"    ROUGE 1-F:{r1['fmeasure']:.4f}  2-F:{r2['fmeasure']:.4f}  "
                  f"L-F:{rL['fmeasure']:.4f}")

        print(f"    ({elapsed:.1f}s)")
        results.append(entry)

    # ── Aggregate ─────────────────────────────────────────────────────────
    n = len(results)
    avg: dict = {}
    if run_bleu and n > 0:
        for k, vals in bleu_accum.items():
            avg[k] = round(sum(vals) / n, 4)
    if run_rouge and n > 0:
        for k, vals in rouge_accum.items():
            avg[k] = round(sum(vals) / n, 4)

    return results, avg


# ═══════════════════════════════════════════════════════════════════════════════
# Retrieval evaluation (Recall@K)
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_retrieval(
    df: pd.DataFrame,
    k_values: list[int],
    store_mode: str = "curriculum",
) -> tuple[list[dict], dict]:
    """Evaluate retrieval quality using Recall@K.

    Returns (per_question_results, aggregate_scores).
    """
    from src.config import settings
    from src.service.vectorstore import curriculum_store, regulation_store

    max_k = max(k_values)
    results: list[dict] = []
    all_recalls: dict[int, list[float]] = {k: [] for k in k_values}

    total = len(df)

    for idx, row in df.iterrows():
        qid = row.get("qid", str(idx))
        question = row["query"]
        gt_ids = extract_gt_chunk_ids(row["retrieval_gt"])

        if not gt_ids:
            print(f"  [{idx + 1}/{total}] SKIP (no retrieval ground truth)")
            continue

        print(f"\n  [{idx + 1}/{total}] Q: {question[:80]}...")
        print(f"    GT IDs: {sorted(gt_ids)}")

        # Determine store
        if store_mode == "auto":
            domain = route_query(question)
        elif store_mode == "both":
            domain = "both"
        else:
            domain = store_mode

        t0 = time.time()
        try:
            if domain == "both":
                c_chunks = curriculum_store.retrieve(question, top_k=max_k)
                r_chunks = regulation_store.retrieve(question, top_k=max_k)
                merged = sorted(c_chunks + r_chunks, key=lambda c: c.score, reverse=True)
                retrieved_ids = [str(c.chunk_id) for c in merged[:max_k]]
            else:
                store = regulation_store if domain == "regulations" else curriculum_store
                chunks = store.retrieve(question, top_k=max_k)
                retrieved_ids = [str(c.chunk_id) for c in chunks]
        except Exception as exc:
            print(f"    ❌ ERROR: {exc}")
            retrieved_ids = []
        elapsed = time.time() - t0

        print(f"    Store: {domain} | Retrieved: {retrieved_ids[:10]}"
              f"{'...' if len(retrieved_ids) > 10 else ''} ({elapsed:.2f}s)")

        recalls: dict[str, float] = {}
        for k in k_values:
            r = compute_recall_at_k(gt_ids, retrieved_ids, k)
            recalls[f"recall@{k}"] = round(r, 4)
            all_recalls[k].append(r)

        recall_parts = [f"R@{k}:{recalls[f'recall@{k}']:.4f}" for k in k_values]
        print(f"    {' '.join(recall_parts)}")

        results.append({
            "qid": qid,
            "question": question,
            "domain": domain,
            "gt_chunk_ids": sorted(gt_ids),
            "retrieved_chunk_ids": retrieved_ids[:max_k],
            "scores": recalls,
            "elapsed_seconds": round(elapsed, 2),
        })

    # ── Aggregate ─────────────────────────────────────────────────────────
    n = len(results)
    avg: dict = {}
    if n > 0:
        for k in k_values:
            avg[f"recall@{k}"] = round(sum(all_recalls[k]) / n, 4)

    return results, avg


# ═══════════════════════════════════════════════════════════════════════════════
# Summary printer
# ═══════════════════════════════════════════════════════════════════════════════

def print_summary(
    mode: str,
    total: int,
    gen_results: list[dict] | None,
    gen_avg: dict | None,
    ret_results: list[dict] | None,
    ret_avg: dict | None,
    k_values: list[int],
):
    """Print a pretty summary table to console."""
    print("\n" + "=" * 64)
    print("  HAUI AGENT — EVALUATION SUMMARY")
    print("=" * 64)

    # ── Generation metrics ────────────────────────────────────────────────
    if gen_results is not None and gen_avg:
        n = len(gen_results)
        print(f"\n  Generation metrics ({n}/{total} questions)")

        if any(k.startswith("bleu") for k in gen_avg):
            print(f"  ┌──────────┬──────────┐")
            print(f"  │  Metric  │  Score   │")
            print(f"  ├──────────┼──────────┤")
            for i in range(1, 5):
                key = f"bleu_{i}"
                if key in gen_avg:
                    print(f"  │  BLEU-{i}  │  {gen_avg[key]:.4f}  │")
            print(f"  └──────────┴──────────┘")

        if any(k.startswith("rouge") for k in gen_avg):
            print(f"  ┌──────────┬───────────┬──────────┬──────────┐")
            print(f"  │  Metric  │ Precision │  Recall  │ F-score  │")
            print(f"  ├──────────┼───────────┼──────────┼──────────┤")
            for m in ["rouge1", "rouge2", "rougeL"]:
                label = {"rouge1": "ROUGE-1", "rouge2": "ROUGE-2", "rougeL": "ROUGE-L"}[m]
                p = gen_avg.get(f"{m}_p", 0)
                r = gen_avg.get(f"{m}_r", 0)
                f = gen_avg.get(f"{m}_f", 0)
                print(f"  │ {label:8s} │  {p:.4f}   │  {r:.4f}  │  {f:.4f}  │")
            print(f"  └──────────┴───────────┴──────────┴──────────┘")

    # ── Retrieval metrics ─────────────────────────────────────────────────
    if ret_results is not None and ret_avg:
        n = len(ret_results)
        print(f"\n  Retrieval metrics ({n}/{total} questions)")
        print(f"  ┌──────────────┬──────────┐")
        print(f"  │    Metric    │  Score   │")
        print(f"  ├──────────────┼──────────┤")
        for k in k_values:
            key = f"recall@{k}"
            if key in ret_avg:
                print(f"  │  Recall@{k:<4d} │  {ret_avg[key]:.4f}  │")
        print(f"  └──────────────┴──────────┘")

    print("=" * 64)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Unified evaluation for HAUI Agent (BLEU + ROUGE + Recall@K)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/evaluate.py --mode all --limit 10
  python scripts/evaluate.py --mode recall --store curriculum --k 1 3 5 10
  python scripts/evaluate.py --mode generation --limit 5
  python scripts/evaluate.py --mode bleu --limit 5
  python scripts/evaluate.py --mode rouge --limit 5
        """,
    )
    parser.add_argument(
        "--mode",
        default="all",
        choices=["all", "generation", "recall", "bleu", "rouge"],
        help="Which metrics to run (default: all)",
    )
    parser.add_argument(
        "--qa",
        default="data/qa.parquet",
        help="Path to the Q&A parquet file (default: data/qa.parquet)",
    )
    parser.add_argument(
        "--output",
        default="evaluation_results.json",
        help="Path to save evaluation results JSON (default: evaluation_results.json)",
    )
    parser.add_argument(
        "--k",
        type=int,
        nargs="+",
        default=[1, 3, 5, 10],
        help="K values for Recall@K (default: 1 3 5 10)",
    )
    parser.add_argument(
        "--store",
        default="curriculum",
        choices=["curriculum", "regulations", "both", "auto"],
        help="Which FAISS store for retrieval eval (default: curriculum)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only evaluate the first N questions",
    )
    parser.add_argument(
        "--log-level",
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: WARNING)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    # ── Load data ─────────────────────────────────────────────────────────
    print(f"Loading QA data from '{args.qa}'...")
    df = pd.read_parquet(args.qa)
    print(f"Loaded {len(df)} question-answer pairs.")

    if args.limit:
        df = df.head(args.limit)
        print(f"Limiting to first {args.limit} questions.")

    total = len(df)
    mode = args.mode

    run_bleu = mode in ("all", "generation", "bleu")
    run_rouge = mode in ("all", "generation", "rouge")
    run_recall = mode in ("all", "recall")

    gen_results, gen_avg = None, None
    ret_results, ret_avg = None, None

    # ── Generation evaluation ─────────────────────────────────────────────
    if run_bleu or run_rouge:
        print(f"\n{'─' * 64}")
        metrics = []
        if run_bleu:
            metrics.append("BLEU")
        if run_rouge:
            metrics.append("ROUGE")
        print(f"  Running generation evaluation: {' + '.join(metrics)}")
        print(f"{'─' * 64}")
        gen_results, gen_avg = evaluate_generation(df, run_bleu=run_bleu, run_rouge=run_rouge)

    # ── Retrieval evaluation ──────────────────────────────────────────────
    if run_recall:
        print(f"\n{'─' * 64}")
        print(f"  Running retrieval evaluation: Recall@K (store={args.store})")
        print(f"{'─' * 64}")
        ret_results, ret_avg = evaluate_retrieval(df, args.k, args.store)

    # ── Summary ───────────────────────────────────────────────────────────
    print_summary(mode, total, gen_results, gen_avg, ret_results, ret_avg, args.k)

    # ── Save JSON ─────────────────────────────────────────────────────────
    output = {
        "evaluation_date": datetime.now().isoformat(),
        "mode": mode,
        "qa_file": args.qa,
        "total_questions": total,
    }

    if gen_results is not None:
        output["generation"] = {
            "evaluated": len(gen_results),
            "average_scores": gen_avg,
            "details": gen_results,
        }

    if ret_results is not None:
        output["retrieval"] = {
            "evaluated": len(ret_results),
            "k_values": args.k,
            "store": args.store,
            "average_scores": ret_avg,
            "details": ret_results,
        }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to '{args.output}'")


if __name__ == "__main__":
    main()
