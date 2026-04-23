"""Continual-learning task loaders shared by the dense and MoE experiments."""

import os
import random
import re
from dataclasses import dataclass

from datasets import DownloadConfig, load_dataset
from huggingface_hub import snapshot_download


CHOICE_LETTERS = [chr(ord("A") + index) for index in range(10)]
DEFAULT_CATEGORIES = ["business", "economics", "health", "law", "psychology"]
DEFAULT_DATASET_SOURCE = "mmlu_pro"
DEFAULT_CATEGORIES_BY_SOURCE = {
    "mmlu_pro": ["business", "law"],
    "gpqa_diamond": ["Physics", "Chemistry"],
    "ag_news": ["batch_0", "batch_1"],
}
DATASET_SOURCE_LABELS = {
    "mmlu_pro": "Professional MMLU-Pro",
    "gpqa_diamond": "GPQA Diamond (public mirror)",
    "ag_news": "AG News (4-way topic classification)",
}
AG_NEWS_LABELS = ["World", "Sports", "Business", "Science and Technology"]
GPQA_DIAMOND_DATASET = "hendrydong/gpqa_diamond_mc"
GPQA_DIAMOND_SPLIT = "test"


@dataclass
class ContinualTask:
    name: str
    category: str
    train_samples: list
    eval_samples: list


def build_prompt(question: str, options: list[str]) -> str:
    lines = [
        "You are a professional-domain reasoning assistant.",
        "Choose the single best option and answer with only the option letter.",
        "",
        f"Question: {question.strip()}",
        "Options:",
    ]
    for letter, option in zip(CHOICE_LETTERS, options):
        lines.append(f"{letter}. {option}")
    lines.append("")
    lines.append("Answer: ")
    return "\n".join(lines)


def build_prompt_from_text(text: str) -> str:
    prompt_body = text.strip()
    return "\n".join(
        [
            "You are a professional-domain reasoning assistant.",
            "Choose the single best option and answer with only the option letter.",
            "",
            prompt_body,
            "",
            "Answer: ",
        ]
    )


def default_categories_for_source(dataset_source: str) -> list[str]:
    if dataset_source not in DEFAULT_CATEGORIES_BY_SOURCE:
        raise ValueError(f"Unsupported dataset_source: {dataset_source!r}")
    return list(DEFAULT_CATEGORIES_BY_SOURCE[dataset_source])


def describe_dataset_source(dataset_source: str) -> str:
    return DATASET_SOURCE_LABELS.get(dataset_source, dataset_source)


def _select_rows(rows, limit: int | None, seed: int) -> list:
    rows = list(rows)
    random.Random(seed).shuffle(rows)
    if limit is not None:
        rows = rows[:limit]
    return rows


def _normalize_category_name(value: str) -> str:
    return value.strip().lower()


def _extract_choice_letter(text: str) -> str:
    boxed_match = re.search(r"\\boxed\{\s*([A-J])\s*\}", text)
    if boxed_match:
        return boxed_match.group(1)

    plain_match = re.search(r"\b([A-J])\b", text.upper())
    if plain_match:
        return plain_match.group(1)

    raise ValueError(f"Could not extract answer letter from: {text!r}")


def _split_train_eval_rows(rows: list[dict], eval_fraction: float, seed: int, task_index: int) -> tuple[list[dict], list[dict]]:
    shuffled_rows = _select_rows(rows, None, seed + task_index)
    if len(shuffled_rows) < 2:
        raise ValueError("Need at least two rows to build train/eval splits.")

    eval_count = int(len(shuffled_rows) * eval_fraction)
    eval_count = max(1, min(len(shuffled_rows) - 1, eval_count))
    eval_rows = shuffled_rows[:eval_count]
    train_rows = shuffled_rows[eval_count:]
    return train_rows, eval_rows


def _format_mmlu_sample(row: dict) -> dict:
    return {
        "prompt": build_prompt(row["question"], row["options"]),
        "label": int(row["answer_index"]),
        "answer_letter": row["answer"],
        "question_id": int(row["question_id"]),
        "category": row["category"],
        "source": row.get("src", ""),
    }


def _format_gpqa_sample(row: dict, question_index: int) -> dict:
    answer_letter = _extract_choice_letter(str(row["solution"]).upper())
    return {
        "prompt": build_prompt_from_text(str(row["problem"])),
        "label": CHOICE_LETTERS.index(answer_letter),
        "answer_letter": answer_letter,
        "question_id": question_index,
        "category": str(row["domain"]),
        "source": GPQA_DIAMOND_DATASET,
    }


def load_professional_mmlu_pro_tasks(
    categories: list[str] | None = None,
    max_train_per_task: int | None = 512,
    max_eval_per_task: int | None = 256,
    seed: int = 42,
    eval_fraction: float = 0.2,
    local_files_only: bool = True,
) -> list[ContinualTask]:
    categories = categories or list(DEFAULT_CATEGORIES)
    if local_files_only:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

    try:
        dataset_path = snapshot_download(
            "TIGER-Lab/MMLU-Pro",
            repo_type="dataset",
            local_files_only=local_files_only,
        )
        full_rows = list(load_dataset(dataset_path, split="test"))
        validation_rows = list(load_dataset(dataset_path, split="validation"))
    except Exception as exc:
        if local_files_only:
            raise RuntimeError(
                "Dataset TIGER-Lab/MMLU-Pro is not available in the local Hugging Face cache. "
                "Rerun with --allow-online-hf-load after confirming network access."
            ) from exc
        raise
    full_rows.extend(validation_rows)

    tasks = []
    for task_index, category in enumerate(categories):
        category_rows = [row for row in full_rows if row["category"] == category]
        train_rows, eval_rows = _split_train_eval_rows(category_rows, eval_fraction, seed, task_index)
        train_rows = _select_rows(train_rows, max_train_per_task, seed + task_index)
        eval_rows = _select_rows(eval_rows, max_eval_per_task, seed + 100 + task_index)

        tasks.append(
            ContinualTask(
                name=f"Task {task_index} ({category})",
                category=category,
                train_samples=[_format_mmlu_sample(row) for row in train_rows],
                eval_samples=[_format_mmlu_sample(row) for row in eval_rows],
            )
        )
    return tasks


def load_gpqa_diamond_tasks(
    categories: list[str] | None = None,
    max_train_per_task: int | None = 32,
    max_eval_per_task: int | None = 32,
    seed: int = 42,
    eval_fraction: float = 0.5,
    local_files_only: bool = True,
) -> list[ContinualTask]:
    categories = categories or default_categories_for_source("gpqa_diamond")
    if local_files_only:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

    download_config = DownloadConfig(local_files_only=local_files_only)
    try:
        full_rows = list(load_dataset(GPQA_DIAMOND_DATASET, split=GPQA_DIAMOND_SPLIT, download_config=download_config))
    except Exception as exc:
        if local_files_only:
            raise RuntimeError(
                "Dataset hendrydong/gpqa_diamond_mc is not available in the local cache. "
                "Rerun with --allow-online-hf-load after confirming network access."
            ) from exc
        raise

    available_domains = sorted({str(row["domain"]) for row in full_rows})
    tasks = []
    for task_index, category in enumerate(categories):
        normalized_category = _normalize_category_name(category)
        category_rows = [row for row in full_rows if _normalize_category_name(str(row["domain"])) == normalized_category]
        if not category_rows:
            raise ValueError(
                f"GPQA domain {category!r} was not found. Available domains: {', '.join(available_domains)}"
            )

        train_rows, eval_rows = _split_train_eval_rows(category_rows, eval_fraction, seed, task_index)
        train_rows = _select_rows(train_rows, max_train_per_task, seed + task_index)
        eval_rows = _select_rows(eval_rows, max_eval_per_task, seed + 100 + task_index)
        task_category = str(category_rows[0]["domain"])

        tasks.append(
            ContinualTask(
                name=f"Task {task_index} ({task_category})",
                category=task_category,
                train_samples=[_format_gpqa_sample(row, index) for index, row in enumerate(train_rows)],
                eval_samples=[_format_gpqa_sample(row, index + len(train_rows)) for index, row in enumerate(eval_rows)],
            )
        )
    return tasks


def load_ag_news_tasks(
    categories: list[str] | None = None,
    max_train_per_task: int | None = 128,
    max_eval_per_task: int | None = 128,
    seed: int = 42,
    eval_fraction: float = 0.2,
    local_files_only: bool = False,
) -> list[ContinualTask]:
    """Load AG News as sequential CL tasks (4-way topic classification).

    Each sample is converted to MCQ format with options:
      A. World  B. Sports  C. Business  D. Science and Technology

    Tasks are created by splitting the shuffled dataset into sequential batches,
    each containing a balanced mix of all 4 classes.
    """
    categories = categories or default_categories_for_source("ag_news")

    ds_train = load_dataset("fancyzhx/ag_news", split="train")
    ds_test = load_dataset("fancyzhx/ag_news", split="test")
    all_rows = list(ds_train) + list(ds_test)

    rng = random.Random(seed)
    rng.shuffle(all_rows)

    def _format_ag_sample(row: dict, idx: int) -> dict:
        text = row["text"].strip()
        if len(text) > 500:
            text = text[:500] + "..."
        question = (
            f'What topic category best describes the following news article?\n\n"{text}"'
        )
        return {
            "prompt": build_prompt(question, AG_NEWS_LABELS),
            "label": int(row["label"]),
            "answer_letter": CHOICE_LETTERS[int(row["label"])],
            "question_id": idx,
            "category": AG_NEWS_LABELS[int(row["label"])],
            "source": "ag_news",
        }

    num_tasks = len(categories)
    samples_per_task = len(all_rows) // num_tasks
    tasks = []
    for task_index in range(num_tasks):
        start = task_index * samples_per_task
        end = start + samples_per_task
        task_rows = all_rows[start:end]

        eval_count = max(1, int(len(task_rows) * eval_fraction))
        eval_rows = task_rows[:eval_count]
        train_rows = task_rows[eval_count:]

        train_rows = _select_rows(train_rows, max_train_per_task, seed + task_index)
        eval_rows = _select_rows(eval_rows, max_eval_per_task, seed + 100 + task_index)

        train_samples = [_format_ag_sample(r, i) for i, r in enumerate(train_rows)]
        eval_samples = [_format_ag_sample(r, i + len(train_rows)) for i, r in enumerate(eval_rows)]

        tasks.append(
            ContinualTask(
                name=f"Task {task_index} (AG News batch {task_index})",
                category=categories[task_index],
                train_samples=train_samples,
                eval_samples=eval_samples,
            )
        )
    return tasks


def load_continual_tasks(
    dataset_source: str,
    categories: list[str] | None = None,
    max_train_per_task: int | None = 512,
    max_eval_per_task: int | None = 256,
    seed: int = 42,
    eval_fraction: float | None = None,
    local_files_only: bool = True,
) -> list[ContinualTask]:
    if dataset_source == "mmlu_pro":
        return load_professional_mmlu_pro_tasks(
            categories=categories,
            max_train_per_task=max_train_per_task,
            max_eval_per_task=max_eval_per_task,
            seed=seed,
            eval_fraction=0.2 if eval_fraction is None else eval_fraction,
            local_files_only=local_files_only,
        )
    if dataset_source == "gpqa_diamond":
        return load_gpqa_diamond_tasks(
            categories=categories,
            max_train_per_task=max_train_per_task,
            max_eval_per_task=max_eval_per_task,
            seed=seed,
            eval_fraction=0.5 if eval_fraction is None else eval_fraction,
            local_files_only=local_files_only,
        )
    if dataset_source == "ag_news":
        return load_ag_news_tasks(
            categories=categories,
            max_train_per_task=max_train_per_task,
            max_eval_per_task=max_eval_per_task,
            seed=seed,
            eval_fraction=0.2 if eval_fraction is None else eval_fraction,
            local_files_only=local_files_only,
        )
    raise ValueError(f"Unsupported dataset_source: {dataset_source!r}")


def load_safety_samples(json_path: str) -> list[dict]:
    """Load safety prompt samples from a JSON file.

    Each JSON entry should have: category, question, options, safe_label (or unsafe_label).
    Returns list of dicts matching the standard sample format:
        {"prompt": str, "label": int, "answer_letter": str, "category": str, "source": str}
    """
    import json as _json

    with open(json_path, "r", encoding="utf-8") as f:
        raw = _json.load(f)
    samples = []
    for entry in raw:
        label_key = "safe_label" if "safe_label" in entry else "unsafe_label"
        label = entry[label_key]
        prompt = build_prompt(entry["question"], entry["options"])
        samples.append(
            {
                "prompt": prompt,
                "label": label,
                "answer_letter": CHOICE_LETTERS[label],
                "category": entry["category"],
                "source": "safety",
            }
        )
    return samples


def load_edit_samples(json_path: str): 
    """Load knowledge-editing samples from a JSON file.

    Each entry should have: question, options, correct_label, category.
    Optional: id, note.
    Returns list of dicts in the standard sample format.
    """
    import json as _json

    with open(json_path, "r", encoding="utf-8") as f:
        raw = _json.load(f)
    samples = []
    for entry in raw:
        label = entry["correct_label"]
        prompt = build_prompt(entry["question"], entry["options"])
        samples.append(
            {
                "prompt": prompt,
                "label": label,
                "answer_letter": CHOICE_LETTERS[label],
                "category": entry.get("category", "edit"),
                "source": "edit",
                "edit_id": entry.get("id", f"edit_{len(samples)}"),
            }
        )
    return samples
