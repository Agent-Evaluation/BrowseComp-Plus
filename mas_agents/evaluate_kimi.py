"""
LLM-as-Judge evaluator using Kimi K2.5 via NVIDIA NIM endpoint.

Adapted from scripts_evaluation/evaluate_with_openai.py.
Uses the OpenAI-compatible NVIDIA NIM API to call Kimi K2.5 as judge.

Usage:
    uv run python -m mas_agents.evaluate_kimi \
        --input-dir runs/bm25/single_gpt5mini \
        --api-key nvapi-YOUR_KEY_HERE

    # Or set NVIDIA_API_KEY environment variable
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import openai
from tqdm import tqdm

from .prompts import GRADER_TEMPLATE

NVIDIA_NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_MODEL = "moonshotai/kimi-k2.5"


def load_ground_truth(jsonl_path: Path) -> Dict[str, Dict[str, str]]:
    """Load ground truth from the decrypted dataset."""
    gt: Dict[str, Dict[str, str]] = {}
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            gt[str(obj["query_id"])] = {
                "question": obj["query"],
                "answer": obj["answer"],
            }
    return gt


def create_judge_prompt(question: str, response: str, correct_answer: str) -> str:
    """Format the grader template with the question, response, and correct answer."""
    return GRADER_TEMPLATE.format(
        question=question, response=response, correct_answer=correct_answer
    )


def call_kimi_judge(
    client: openai.OpenAI,
    prompt: str,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 4096,
) -> str:
    """Call Kimi K2.5 via NVIDIA NIM to judge a response."""
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=1.0,
        top_p=1.0,
        extra_body={"chat_template_kwargs": {"thinking": True}},
    )

    if response.choices and response.choices[0].message.content:
        return response.choices[0].message.content.strip()
    return ""


def parse_judge_response(judge_response: str) -> dict:
    """Parse the structured judge response into fields."""
    result = {
        "extracted_final_answer": None,
        "reasoning": None,
        "correct": None,
        "confidence": None,
        "parse_error": False,
    }

    if not judge_response:
        result["parse_error"] = True
        return result

    # Extract extracted_final_answer (try bold formats first, then regular)
    answer_match = re.search(
        r"\*\*extracted_final_answer:\*\*\s*(.*?)(?=\n|$)",
        judge_response, re.IGNORECASE | re.DOTALL,
    )
    if not answer_match:
        answer_match = re.search(
            r"\*\*extracted_final_answer\*\*:\s*(.*?)(?=\n|$)",
            judge_response, re.IGNORECASE | re.DOTALL,
        )
    if not answer_match:
        answer_match = re.search(
            r"extracted_final_answer:\s*(.*?)(?=\n|$)",
            judge_response, re.IGNORECASE | re.DOTALL,
        )
    if answer_match:
        result["extracted_final_answer"] = answer_match.group(1).strip()

    # Extract reasoning
    reasoning_match = re.search(
        r"\*\*reasoning:\*\*\s*(.*?)(?=\n\*\*correct|\ncorrect:|$)",
        judge_response, re.IGNORECASE | re.DOTALL,
    )
    if not reasoning_match:
        reasoning_match = re.search(
            r"reasoning:\s*(.*?)(?=\n\*\*correct|\ncorrect:|$)",
            judge_response, re.IGNORECASE | re.DOTALL,
        )
    if reasoning_match:
        result["reasoning"] = reasoning_match.group(1).strip()

    # Extract correct (yes/no)
    correct_match = re.search(
        r"\*\*correct:\*\*\s*(yes|no)",
        judge_response, re.IGNORECASE,
    )
    if not correct_match:
        correct_match = re.search(
            r"correct:\s*(yes|no)",
            judge_response, re.IGNORECASE,
        )
    if correct_match:
        result["correct"] = correct_match.group(1).strip().lower()
    else:
        result["parse_error"] = True

    # Extract confidence
    conf_match = re.search(
        r"\*\*confidence:\*\*\s*(\d+)",
        judge_response, re.IGNORECASE,
    )
    if not conf_match:
        conf_match = re.search(
            r"confidence:\s*(\d+)",
            judge_response, re.IGNORECASE,
        )
    if conf_match:
        result["confidence"] = int(conf_match.group(1))

    return result


def extract_final_output(result_data: dict) -> str:
    """Extract the final output text from a run result JSON."""
    for item in reversed(result_data.get("result", [])):
        if item.get("type") == "output_text" and item.get("output"):
            return item["output"]
    return ""


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate BrowseComp-Plus run results using Kimi K2.5 as judge"
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing run_*.json result files",
    )
    parser.add_argument(
        "--ground-truth",
        default="data/browsecomp_plus_decrypted.jsonl",
        help="Path to decrypted ground truth JSONL (default: %(default)s)",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="NVIDIA NIM API key (or set NVIDIA_API_KEY env var)",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Judge model (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--output-file",
        default=None,
        help="Path to save evaluation summary JSON (default: <input_dir>/eval_summary.json)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Max retries per judge call (default: %(default)s)",
    )

    args = parser.parse_args()

    # ── API key ───────────────────────────────────────────────────────────────
    api_key = args.api_key or os.getenv("NVIDIA_API_KEY")
    if not api_key:
        print("Error: NVIDIA API key required. Use --api-key or set NVIDIA_API_KEY env var.")
        sys.exit(1)

    client = openai.OpenAI(base_url=NVIDIA_NIM_BASE_URL, api_key=api_key)

    # ── Load ground truth ─────────────────────────────────────────────────────
    gt_path = Path(args.ground_truth)
    if not gt_path.is_file():
        print(f"Error: Ground truth file not found: {gt_path}")
        sys.exit(1)

    gt = load_ground_truth(gt_path)
    print(f"Loaded {len(gt)} ground truth entries")

    # ── Load run results ──────────────────────────────────────────────────────
    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        print(f"Error: Input directory not found: {input_dir}")
        sys.exit(1)

    run_files = sorted(input_dir.glob("run_*.json"))
    print(f"Found {len(run_files)} result files in {input_dir}")

    # ── Evaluate each result ──────────────────────────────────────────────────
    per_query_metrics = []
    correct_count = 0
    total_evaluated = 0
    confidences = []
    correctness_list = []

    for rf in tqdm(run_files, desc="Evaluating", unit="query"):
        try:
            with rf.open("r", encoding="utf-8") as f:
                run_data = json.load(f)
        except Exception as e:
            print(f"  [warn] Could not read {rf.name}: {e}")
            continue

        qid = str(run_data.get("query_id", ""))
        status = run_data.get("status", "")

        if qid not in gt:
            print(f"  [warn] Query ID {qid} not found in ground truth, skipping")
            continue

        # Failed runs count as incorrect
        if status != "completed":
            per_query_metrics.append({
                "query_id": qid,
                "status": status,
                "correct": "no",
                "confidence": 0,
                "reasoning": "Run did not complete",
            })
            correctness_list.append(0)
            confidences.append(0)
            total_evaluated += 1
            continue

        # Extract agent's final output
        agent_output = extract_final_output(run_data)
        if not agent_output:
            per_query_metrics.append({
                "query_id": qid,
                "status": "no_output",
                "correct": "no",
                "confidence": 0,
                "reasoning": "No output text found in result",
            })
            correctness_list.append(0)
            confidences.append(0)
            total_evaluated += 1
            continue

        # Call judge
        question = gt[qid]["question"]
        correct_answer = gt[qid]["answer"]
        judge_prompt = create_judge_prompt(question, agent_output, correct_answer)

        judge_response = ""
        for attempt in range(args.max_retries):
            try:
                judge_response = call_kimi_judge(client, judge_prompt, args.model)
                break
            except Exception as e:
                if attempt < args.max_retries - 1:
                    print(f"  [retry] Judge call for {qid} failed: {e}")
                else:
                    print(f"  [error] Judge call for {qid} failed after {args.max_retries} attempts: {e}")

        parsed = parse_judge_response(judge_response)

        is_correct = parsed.get("correct") == "yes"
        if is_correct:
            correct_count += 1
            correctness_list.append(1)
        else:
            correctness_list.append(0)

        confidence = parsed.get("confidence", 100) or 100
        confidences.append(confidence)
        total_evaluated += 1

        per_query_metrics.append({
            "query_id": qid,
            "status": "evaluated",
            "correct": parsed.get("correct", "unknown"),
            "confidence": confidence,
            "extracted_answer": parsed.get("extracted_final_answer"),
            "reasoning": parsed.get("reasoning"),
            "parse_error": parsed.get("parse_error", False),
            "tool_calls": run_data.get("tool_call_counts", {}),
        })

    # ── Compute aggregate metrics ─────────────────────────────────────────────
    accuracy = correct_count / total_evaluated if total_evaluated > 0 else 0

    # Calibration error: |mean_confidence - accuracy|
    mean_confidence = np.mean(confidences) / 100.0 if confidences else 0
    calibration_error = abs(mean_confidence - accuracy) * 100

    # Average tool stats
    avg_searches = np.mean([
        m.get("tool_calls", {}).get("search", 0)
        for m in per_query_metrics
        if m.get("tool_calls")
    ]) if per_query_metrics else 0

    # ── Build summary ─────────────────────────────────────────────────────────
    summary = {
        "LLM": args.input_dir,
        "Judge": args.model,
        "Accuracy (%)": round(accuracy * 100, 2),
        "Total Evaluated": total_evaluated,
        "Correct": correct_count,
        "avg_tool_stats": {"search": round(float(avg_searches), 2)},
        "Calibration Error (%)": round(calibration_error, 2),
        "Retriever": "BM25",
        "Evaluation Date": datetime.now().strftime("%Y-%m-%d"),
        "per_query_metrics": per_query_metrics,
    }

    # ── Save ──────────────────────────────────────────────────────────────────
    output_file = args.output_file or str(input_dir / "eval_summary.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n{'=' * 60}")
    print(f"Evaluation Complete")
    print(f"  Accuracy:          {accuracy:.1%} ({correct_count}/{total_evaluated})")
    print(f"  Calibration Error: {calibration_error:.2f}%")
    print(f"  Avg Searches:      {avg_searches:.1f}")
    print(f"  Summary saved to:  {output_file}")


if __name__ == "__main__":
    main()
