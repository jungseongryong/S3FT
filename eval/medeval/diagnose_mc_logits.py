#!/usr/bin/env python3
import argparse
import json
import os
from collections import Counter, defaultdict
from typing import Any, Dict, List

from vllm import LLM, SamplingParams

from run_med_eval import (
    extract_answer_letter_fallback,
    format_question_medmcqa,
    format_question_medqa,
    format_question_mmlu,
    get_correct_answer,
    load_jsonl,
)


def format_prompt(item: Dict[str, Any], dataset_type: str) -> str:
    if dataset_type == "medqa":
        return format_question_medqa(item)
    if dataset_type == "mmlu":
        return format_question_mmlu(item)
    return format_question_medmcqa(item)


def choice_logprobs(output, llm) -> Dict[str, float]:
    if (
        not hasattr(output.outputs[0], "logprobs")
        or output.outputs[0].logprobs is None
        or len(output.outputs[0].logprobs) == 0
    ):
        fallback = extract_answer_letter_fallback(output.outputs[0].text)
        return {fallback: 0.0}

    tokenizer = llm.get_tokenizer()
    scores = {}
    for token_id, logprob_data in output.outputs[0].logprobs[0].items():
        token = tokenizer.decode([token_id]).strip().upper()
        if token in ["A", "B", "C", "D"]:
            scores[token] = float(logprob_data.logprob)
    return scores


def summarize_dataset(llm: LLM, data: List[Dict[str, Any]], dataset_type: str, max_samples: int):
    if max_samples > 0:
        data = data[:max_samples]

    prompts = [format_prompt(item, dataset_type) for item in data]
    outputs = llm.generate(prompts, SamplingParams(temperature=0.0, top_p=1.0, max_tokens=1, logprobs=20))

    pred_counts = Counter()
    gold_counts = Counter()
    correct = 0
    margins = []
    missing_choice_logprobs = 0
    examples = []

    for item, output in zip(data, outputs):
        scores = choice_logprobs(output, llm)
        if scores:
            pred = max(scores, key=scores.get)
        else:
            pred = extract_answer_letter_fallback(output.outputs[0].text)
            missing_choice_logprobs += 1

        gold = get_correct_answer(item, dataset_type)
        pred_counts[pred] += 1
        gold_counts[gold] += 1
        correct += int(pred == gold)

        sorted_scores = sorted(scores.values(), reverse=True)
        if len(sorted_scores) >= 2:
            margins.append(sorted_scores[0] - sorted_scores[1])

        if len(examples) < 20:
            examples.append(
                {
                    "pred": pred,
                    "gold": gold,
                    "correct": pred == gold,
                    "scores": scores,
                    "text": output.outputs[0].text,
                }
            )

    total = len(data)
    return {
        "accuracy": correct / total if total else 0.0,
        "total_samples": total,
        "pred_counts": dict(pred_counts),
        "gold_counts": dict(gold_counts),
        "pred_rates": {k: v / total for k, v in pred_counts.items()},
        "gold_rates": {k: v / total for k, v in gold_counts.items()},
        "mean_top2_margin": sum(margins) / len(margins) if margins else None,
        "missing_choice_logprobs": missing_choice_logprobs,
        "examples": examples,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--test_data_dir", default="test_data")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    args = parser.parse_args()

    datasets = {
        "medqa": os.path.join(args.test_data_dir, "medqa_test.jsonl"),
        "mmlu": os.path.join(args.test_data_dir, "mmlu_medical_test.jsonl"),
        "medmcqa": os.path.join(args.test_data_dir, "medmcqa_test.jsonl"),
    }

    llm = LLM(model=args.model, tensor_parallel_size=args.tensor_parallel_size)
    result = {"model": args.model, "datasets": {}}
    try:
        for dataset_type, path in datasets.items():
            data = load_jsonl(path)
            result["datasets"][dataset_type] = summarize_dataset(llm, data, dataset_type, args.max_samples)
            print(dataset_type, result["datasets"][dataset_type]["accuracy"], result["datasets"][dataset_type]["pred_counts"])
    finally:
        del llm

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
