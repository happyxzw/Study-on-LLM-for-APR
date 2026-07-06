import argparse
import json
import glob
import os
import pandas as pd
import numpy as np
from collections import Counter,defaultdict
import csv

def load_json_file(filepath):
    """Quick loader for JSON files."""
    with open(filepath) as f:
        return json.load(f)

def load_bug_categories(classification_file: str) -> dict:
    with open(classification_file) as f:
        data = json.load(f)
    categories = {}
    for bug_name, info in data["data"].items():
        if "error" not in info:
            categories[bug_name] = info["category"]
    return categories

def count_correctness(data):
    """Count patch-level correctness per project (bug)."""
    stats = {}
    for project, info in data.items():
        counts = {"plausible": 0, "wrong": 0, "uncompilable": 0, "timeout": 0}
        for patch in info["output"]:
            key = str(patch["correctness"])
            if key in counts:
                counts[key] += 1
        stats[project] = counts
    return stats



def classify_result(row):
    """
    Decide final bug-level label.
    Order matters and follows standard APR evaluation logic.
    """
    if row["plausible"] > 0:
        return "plausible"
    elif row["wrong"] > 0:
        return "wrong"
    elif row["uncompilable"] > 0:
        return "uncompilable"
    elif row["timeout"] > 0:
        return "timeout"
    else:
        return "unknown"


def model_confidence(table, cols):
    """
    Model-level confidence:
    proportion of the most frequent correctness type
    among all generated patches.
    """
    total = table[cols].sum()
    return total.max() / total.sum() if total.sum() > 0 else 0.0


def model_entropy(table, cols):
    """
    Model-level entropy:
    uncertainty of the model's overall patch correctness distribution.
    """
    total = table[cols].sum()
    if total.sum() == 0:
        return 0.0
    p = total / total.sum()
    p = p[p > 0]
    return -np.sum(p * np.log(p))

def process_file(filepath, bug_categories=None):
    data = load_json_file(filepath)["data"]
    stats = count_correctness(data)

    table = pd.DataFrame.from_dict(stats).transpose()
    count_cols = ["plausible", "wrong", "uncompilable", "timeout"]
    table["bug_label"] = table.apply(classify_result, axis=1)

    category_stats = None
    if bug_categories is not None:
        category_stats = defaultdict(
            lambda: {"plausible": 0, "total": 0, "bugs": []}
        )
        
        for bug_name, row in table.iterrows():
            def normalize_key(name):
                return name.upper().replace("-", "_").replace("/", "_")

            bug_key = normalize_key(bug_name)
            category = bug_categories.get(bug_name)

            if category is None:
                category = "Unknown"
            category_stats[category]["total"] += 1
            category_stats[category]["bugs"].append(bug_name)
            if row["bug_label"] == "plausible":
                category_stats[category]["plausible"] += 1

        for cat in category_stats:
            s = category_stats[cat]
            s["repair_rate"] = (
                s["plausible"] / s["total"] if s["total"] > 0 else 0.0
            )

    return os.path.basename(filepath), table, count_cols, category_stats


def append_result_csv(model, dataset, shot, config,
                      plausible, wrong, uncompilable,
                      unknown, timeout, confidence, entropy):

    csv_path = os.path.expanduser("~/peft4apr/results.csv")

    row = {
        "model": model,
        "dataset": dataset,
        "shot": shot,
        "config": config,
        "plausible": plausible,
        "wrong": wrong,
        "uncompilable": uncompilable,
        "unknown": unknown,
        "timeout": timeout,
        "confidence": round(confidence, 3),
        "entropy": round(entropy, 3)
    }

    rows = []

    if os.path.exists(csv_path):
        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

    updated = False

    for r in rows:
        if (r["model"] == model and
            r["dataset"] == dataset and
            r["shot"] == shot and
            r["config"] == config):

            r.update({k: str(v) for k, v in row.items()})
            updated = True
            break

    if not updated:
        rows.append(row)

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        writer.writeheader()
        writer.writerows(rows)
        
def append_category_csv(model, dataset, shot, config, category_stats):
    csv_path = os.path.expanduser("~/peft4apr/results_by_category.csv")
    CATEGORIES = [
        "Expression Bug", "Control Flow Bug",
        "Reference Bug", "Statement Bug"
    ]
    rows = []
    if os.path.exists(csv_path):
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

    for category in CATEGORIES:
        stats = category_stats.get(category, {})
        row = {
            "model": model, "dataset": dataset,
            "shot": shot, "config": config,
            "category": category,
            "plausible": stats.get("plausible", 0),
            "total": stats.get("total", 0),
            "repair_rate": round(stats.get("repair_rate", 0.0), 3)
        }
        updated = False
        for r in rows:
            if (r["model"] == model and r["dataset"] == dataset
                    and r["shot"] == shot and r["config"] == config
                    and r["category"] == category):
                r.update({k: str(v) for k, v in row.items()})
                updated = True
                break
        if not updated:
            rows.append(row)

    fieldnames = [
        "model", "dataset", "shot", "config",
        "category", "plausible", "total", "repair_rate"
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
         
def main():
    parser = argparse.ArgumentParser(description="Summarize APR patch results.")
    parser.add_argument("results_dir", help="Path to validation results dir")
    parser.add_argument("benchmark_name", help="Name of the benchmark")
    parser.add_argument("model_name", help="Name of the model")
    parser.add_argument("category_file",default=os.path.expanduser("~/peft4apr/quixbugs_bug_classification.json"),help="Path to bug classification JSON (optional)")
    args = parser.parse_args()
    bug_categories = None
    if os.path.exists(args.category_file):
        bug_categories = load_bug_categories(args.category_file)
        print(f"Loaded bug categories from {args.category_file}")
    else:
        print("No category file found, skipping category analysis")

    rows = []

    for filepath in glob.glob(f"{args.results_dir}/*_valid*.json"):
        name, table, count_cols, category_stats = process_file(filepath, bug_categories)

        if "c1" in filepath:
            config = "c1"
        elif "c2" in filepath:
            config = "c2"
        else:
            config = "default"

        if "cot" in name:
            strategy = "cot"
        elif "rag" in name:
            strategy = "rag"
        elif "lora" in name:
            strategy = "lora"
        elif "finetune" in name or "ft" in name:
            strategy= "finetune"
        elif "few" in name:
            strategy = "few-shot"
        else:
            strategy = "zero-shot"

        # bug-level counts (must sum to total bugs, e.g., 40)
        bug_counts = Counter(table["bug_label"])

        count_cols = ["plausible", "wrong", "uncompilable", "timeout"]

        row = {
            "file": name,
            "plausible": bug_counts.get("plausible", 0),
            "wrong": bug_counts.get("wrong", 0),
            "uncompilable": bug_counts.get("uncompilable", 0),
            "unknown":bug_counts.get("unknown", 0),
            "timeout": bug_counts.get("timeout", 0),
            "confidence": model_confidence(table, count_cols),
            "entropy": model_entropy(table, count_cols),
        }

        rows.append(row)
        append_result_csv(
            args.model_name,
            args.benchmark_name,
            strategy,
            config,
            row["plausible"],
            row["wrong"],
            row["uncompilable"],
            row["unknown"],
            row["timeout"],
            row["confidence"],
            row["entropy"]
        )
        if category_stats is not None:
            append_category_csv(
                args.model_name, args.benchmark_name,
                strategy, config, category_stats
            )
            print(f"\n=== Category Analysis: {name} ===")
            CATEGORIES = [
                "Expression Bug", "Control Flow Bug",
                "Reference Bug", "Statement Bug"
            ]
            for cat in CATEGORIES:
                s = category_stats.get(cat, {})
                p = s.get("plausible", 0)
                t = s.get("total", 0)
                r = s.get("repair_rate", 0)
                print(f"  {cat:<22} {p}/{t} ({r:.1%})")
    df = pd.DataFrame(rows).set_index("file")

    print(df.to_latex(
        float_format="%.3f",
        caption=f"Bug-level APR results of {args.model_name} on {args.benchmark_name}",
        column_format="lcccccc",
        position="htbp",
        escape=False
    ))


if __name__ == "__main__":
    main()
