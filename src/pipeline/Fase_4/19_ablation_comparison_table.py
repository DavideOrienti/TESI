from __future__ import annotations
import json
import pandas as pd

from src.utils.io import load_settings

VARIANTS = ["full", "no_tags", "no_overview", "genres_only"]
TOP_K_LIST = [5, 10, 20]
METRICS = ["HR", "NDCG", "MRR"]
INPUT_DIR_NAME = "content_ablation_results"
OUTPUT_DIR_NAME = "content_ablation_results"


def load_variant_summary(base, variant: str) -> dict:
    path = base / INPUT_DIR_NAME / f"summary_{variant}.json"
    if not path.exists():
        raise FileNotFoundError(f"summary mancante per variante '{variant}': {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_row(variant: str, split: str, metrics_dict: dict) -> dict:
    row = {"variant": variant, "split": split}
    for metric in METRICS:
        for k in TOP_K_LIST:
            key = f"{metric}@{k}"
            row[key] = metrics_dict.get(key, float("nan"))
    return row


def build_comparison_df(base) -> pd.DataFrame:
    rows = []
    for variant in VARIANTS:
        summary = load_variant_summary(base, variant)
        rows.append(build_row(variant, "validation", summary["validation"]))
        rows.append(build_row(variant, "test", summary["test"]))
    return pd.DataFrame(rows)


def compute_delta_vs_full(df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [f"{m}@{k}" for m in METRICS for k in TOP_K_LIST]
    rows = []
    for split in ["validation", "test"]:
        full_row = df[(df["variant"] == "full") & (df["split"] == split)]
        if full_row.empty:
            continue
        full_vals = full_row.iloc[0][metric_cols]
        for _, row in df[df["split"] == split].iterrows():
            delta_row = {"variant": row["variant"], "split": split}
            for col in metric_cols:
                base_val = full_vals[col]
                delta_row[col] = (
                    (row[col] - base_val) / base_val * 100.0
                    if base_val != 0 else float("nan")
                )
            rows.append(delta_row)
    return pd.DataFrame(rows)


def df_to_markdown(df: pd.DataFrame) -> str:
    metric_cols = [f"{m}@{k}" for m in METRICS for k in TOP_K_LIST]
    header = ["variant", "split"] + metric_cols
    lines = []
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    for _, row in df.iterrows():
        cells = [str(row["variant"]), str(row["split"])]
        for col in metric_cols:
            val = row[col]
            cells.append(f"{val:.4f}" if pd.notna(val) else "N/A")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def print_table(df: pd.DataFrame, title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    metric_cols = [f"{m}@{k}" for m in METRICS for k in TOP_K_LIST]
    pd.set_option("display.float_format", "{:.4f}".format)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    print(df[["variant", "split"] + metric_cols].to_string(index=False))


def main():
    s = load_settings()
    base = s.paths.processed
    output_dir = base / OUTPUT_DIR_NAME
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[19] dataset={s.dataset}")
    print(f"[19] reading from {base / INPUT_DIR_NAME}")

    df = build_comparison_df(base)

    csv_path = output_dir / "ablation_comparison_table.csv"
    md_path = output_dir / "ablation_comparison_table.md"

    df.to_csv(csv_path, index=False)

    md_content = "# Ablation Study — Comparison Table\n\n"
    for split in ["validation", "test"]:
        md_content += f"## {split.capitalize()}\n\n"
        md_content += df_to_markdown(df[df["split"] == split].reset_index(drop=True))
        md_content += "\n\n"

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    print_table(df[df["split"] == "validation"].reset_index(drop=True), "VALIDATION")
    print_table(df[df["split"] == "test"].reset_index(drop=True), "TEST")

    delta_df = compute_delta_vs_full(df)
    print_table(delta_df[delta_df["split"] == "validation"].reset_index(drop=True),
                "DELTA % vs 'full' — VALIDATION")
    print_table(delta_df[delta_df["split"] == "test"].reset_index(drop=True),
                "DELTA % vs 'full' — TEST")

    delta_csv = output_dir / "ablation_delta_vs_full.csv"
    delta_df.to_csv(delta_csv, index=False)

    print(f"\nsaved -> {csv_path}")
    print(f"saved -> {md_path}")
    print(f"saved -> {delta_csv}")


if __name__ == "__main__":
    main()
