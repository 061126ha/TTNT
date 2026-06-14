"""
Unified Evaluation for HAUI Agent — BLEU, ROUGE & Recall@K (FIXED STABLE VERSION)
"""

import argparse
import logging
import sys

import numpy as np
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

from dotenv import load_dotenv

load_dotenv()

# ── UTF-8 FIX ─────────────────────────────
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr.encoding != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8")

logger = logging.getLogger(__name__)
SMOOTHING = SmoothingFunction().method1


# =========================
# TOKENIZER (simple safe)
# =========================
def tokenize_vi(text: str) -> list[str]:
    if not text:
        return []
    return text.lower().strip().split()


class ViTokenizer:
    def tokenize(self, text):
        return tokenize_vi(text)


# =========================
# GT EXTRACTION
# =========================
def extract_references(gt):
    if not gt:
        return []

    if isinstance(gt, np.ndarray):
        gt = gt.tolist()

    if isinstance(gt, str):
        return [gt]

    if isinstance(gt, list):
        out = []
        for x in gt:
            if isinstance(x, list):
                out.extend([str(i).strip() for i in x if i])
            elif x:
                out.append(str(x).strip())
        return out

    return []


def extract_gt_chunk_ids(gt):
    ids = set()

    if not gt:
        return ids

    if isinstance(gt, np.ndarray):
        gt = gt.tolist()

    def add(v):
        if v is None:
            return
        s = str(v).strip()
        if s:
            ids.add(s)

    if isinstance(gt, list):
        for i in gt:
            if isinstance(i, list):
                for j in i:
                    add(j)
            else:
                add(i)
    else:
        add(gt)

    return ids


# =========================
# BLEU
# =========================
def compute_bleu(refs, cand):
    refs = [tokenize_vi(r) for r in refs]
    cand = tokenize_vi(cand)

    if not refs or not cand:
        return {"bleu_1": 0, "bleu_2": 0, "bleu_3": 0, "bleu_4": 0}

    weights = {
        "bleu_1": (1, 0, 0, 0),
        "bleu_2": (0.5, 0.5, 0, 0),
        "bleu_3": (1/3, 1/3, 1/3, 0),
        "bleu_4": (0.25, 0.25, 0.25, 0.25),
    }

    return {
        k: round(
            sentence_bleu(refs, cand, weights=w, smoothing_function=SMOOTHING),
            4
        )
        for k, w in weights.items()
    }


# =========================
# ROUGE
# =========================
def compute_rouge(refs, cand):
    if not refs or not cand:
        empty = {"precision": 0, "recall": 0, "fmeasure": 0}
        return {"rouge1": empty, "rouge2": empty, "rougeL": empty}

    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"],
        tokenizer=ViTokenizer()
    )

    best = {}

    for r in refs:
        scores = scorer.score(r, cand)

        for k, v in scores.items():
            cur = {
                "precision": round(v.precision, 4),
                "recall": round(v.recall, 4),
                "fmeasure": round(v.fmeasure, 4),
            }

            if k not in best or cur["fmeasure"] > best[k]["fmeasure"]:
                best[k] = cur

    return best


# =========================
# RECALL@K
# =========================
def recall_at_k(gt, pred, k):
    if not gt:
        return 0.0

    pred = pred[:k]
    return len(set(pred) & set(gt)) / len(gt)


# =========================
# GENERATION EVAL
# =========================
def evaluate_generation(df, agent):
    results = []
    bleu_acc = {"bleu_1": [], "bleu_2": [], "bleu_3": [], "bleu_4": []}

    for i, row in df.iterrows():
        q = row["query"]
        refs = extract_references(row.get("generation_gt"))

        if not refs:
            continue

        print(f"[GEN {i}] {q[:60]}")

        try:
            resp = agent.chat(q)
            cand = resp.answer or ""
        except Exception as e:
            print("ERROR:", e)
            cand = ""

        bleu = compute_bleu(refs, cand)

        for k in bleu:
            bleu_acc[k].append(bleu[k])

        results.append({
            "query": q,
            "candidate": cand,
            "bleu": bleu
        })

    avg = {
        k: round(sum(v) / len(v), 4) if v else 0
        for k, v in bleu_acc.items()
    }

    return results, avg


# =========================
# RETRIEVAL EVAL
# =========================
def evaluate_retrieval(df, store, k_list):
    results = []
    acc = {k: [] for k in k_list}

    for i, row in df.iterrows():
        q = row["query"]
        gt = extract_gt_chunk_ids(row.get("retrieval_gt"))

        try:
            chunks = store.retrieve(q, top_k=max(k_list))
            pred = [c.chunk_id for c in chunks]
        except Exception as e:
            print("ERROR:", e)
            pred = []

        scores = {}

        for k in k_list:
            r = recall_at_k(gt, pred, k)
            scores[f"recall@{k}"] = round(r, 4)
            acc[k].append(r)

        results.append({
            "query": q,
            "scores": scores
        })

    avg = {
        f"recall@{k}": round(sum(v) / len(v), 4) if v else 0
        for k, v in acc.items()
    }

    return results, avg


# =========================
# MAIN
# =========================
def main():
    from src.agent.haui_agent import HAUIAgent
    from src.service.vectorstore import curriculum_store

    parser = argparse.ArgumentParser()
    parser.add_argument("--qa", default="data/qa.parquet")
    parser.add_argument("--mode", default="all")
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()

    df = pd.read_parquet(args.qa).head(args.limit)

    agent = HAUIAgent()

    gen_avg, ret_avg = None, None

    if args.mode in ["all", "generation"]:
        print("\n=== GENERATION ===")
        _, gen_avg = evaluate_generation(df, agent)

    if args.mode in ["all", "recall"]:
        print("\n=== RETRIEVAL ===")
        _, ret_avg = evaluate_retrieval(df, curriculum_store, [1, 3, 5, 10])

    print("\n========== RESULT ==========")
    print("GEN:", gen_avg)
    print("RET:", ret_avg)


if __name__ == "__main__":
    main()