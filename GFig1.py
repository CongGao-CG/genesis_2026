from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(SCRIPT_DIR / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(SCRIPT_DIR / ".cache"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_INPUT = Path("g1_tc_by_basin_model_experiment_year.tsv")
DEFAULT_OUTPUT = Path("GFig1.pdf")
BASIN_ORDER = ["AS", "BoB", "WNP", "ENP", "NA", "SI", "SP"]
BASELINE_EXPERIMENT = "historical"
BASELINE_YEARS = (1980, 2014)
SCENARIO_ORDER = ["ssp245", "ssp585"]
GROUP_ORDER = ["Global", "Northern Hemisphere", "Southern Hemisphere"]
GROUP_BASINS = {
    "Global": BASIN_ORDER,
    "Northern Hemisphere": ["AS", "BoB", "WNP", "ENP", "NA"],
    "Southern Hemisphere": ["SI", "SP"],
}
WINDOWS = [
    ("a", "Near-term (2015–2049)", (2015, 2049)),
    ("b", "Mid-century (2040–2074)", (2040, 2074)),
    ("c", "Late-century (2065–2099)", (2065, 2099)),
]
DIFF_PANEL = ("d", "Late-century SSP585 − SSP245")
DIFF_EXPERIMENT = "ssp585_minus_ssp245"
EXPERIMENT_LABELS = {
    "ssp245": "SSP245",
    "ssp585": "SSP585",
}
EXPERIMENT_COLORS = {"ssp245": "#E69F00", "ssp585": "#E45756"}
DIFF_COLOR = "#7A7A7A"
CAPSIZE = 9


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot global and hemispheric changes in TC frequency relative to "
            "historical for future windows from "
            "g1_tc_by_basin_model_experiment_year.tsv."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Input TSV file (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output figure path (default: {DEFAULT_OUTPUT})",
    )
    return parser.parse_args()


def format_entries(entries: list[tuple], limit: int = 6) -> str:
    preview = ", ".join(str(entry) for entry in entries[:limit])
    if len(entries) > limit:
        preview += f", ... (+{len(entries) - limit} more)"
    return preview


def load_rate_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", keep_default_na=False)
    required = {
        "basin",
        "model",
        "experiment",
        "year",
        "rate_per_year_per_realization",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {path}: {sorted(missing)}")

    missing_experiments = {BASELINE_EXPERIMENT, *SCENARIO_ORDER} - set(df["experiment"])
    if missing_experiments:
        raise ValueError(
            f"Missing required experiments in {path}: {sorted(missing_experiments)}"
        )

    missing_basins = set(BASIN_ORDER) - set(df["basin"])
    if missing_basins:
        raise ValueError(f"Missing required basins in {path}: {sorted(missing_basins)}")

    return df


def aggregate_period_rates(
    df: pd.DataFrame, experiment: str, year_start: int, year_end: int
) -> pd.Series:
    models = sorted(df["model"].drop_duplicates().tolist())
    years = list(range(year_start, year_end + 1))
    subset = df[
        (df["experiment"] == experiment) & df["year"].between(year_start, year_end)
    ].copy()

    duplicate_cols = ["basin", "model", "year"]
    duplicates = subset[subset.duplicated(duplicate_cols, keep=False)][duplicate_cols]
    if not duplicates.empty:
        entries = list(duplicates.drop_duplicates().itertuples(index=False, name=None))
        raise ValueError(
            f"Duplicate basin/model/year rows found for {experiment}: "
            f"{format_entries(entries)}"
        )

    series = subset.set_index(["basin", "model", "year"])["rate_per_year_per_realization"]
    expected_index = pd.MultiIndex.from_product(
        [BASIN_ORDER, models, years], names=["basin", "model", "year"]
    )
    missing_entries = expected_index.difference(series.index)
    if len(missing_entries) > 0:
        raise ValueError(
            f"Missing basin/model/year rows for {experiment} {year_start}-{year_end}: "
            f"{format_entries(missing_entries.tolist())}"
        )

    series = series.reindex(expected_index)
    return series.groupby(level=["basin", "model"]).mean()


def summarize_series(changes: pd.Series) -> tuple[float, float]:
    return float(changes.mean()), float(changes.sem())


def compute_group_percent_changes(
    historical_rates: pd.Series, future_rates: pd.Series, group_name: str, group_basins: list[str]
) -> pd.Series:
    historical_basin_index = historical_rates.index.get_level_values("basin")
    future_basin_index = future_rates.index.get_level_values("basin")
    historical_group = (
        historical_rates[historical_basin_index.isin(group_basins)]
        .groupby(level="model")
        .sum()
    )
    future_group = (
        future_rates[future_basin_index.isin(group_basins)]
        .groupby(level="model")
        .sum()
    )

    zero_baseline = historical_group.index[historical_group == 0].tolist()
    if zero_baseline:
        raise ValueError(
            f"Historical {group_name} rate is zero for models, so percent "
            f"change is undefined: {format_entries([(model,) for model in zero_baseline])}"
        )

    return (future_group - historical_group).div(historical_group).mul(100.0)


def compute_change_summary(df: pd.DataFrame) -> pd.DataFrame:
    historical_rates = aggregate_period_rates(df, BASELINE_EXPERIMENT, *BASELINE_YEARS)
    rows: list[dict[str, object]] = []

    for panel_label, window_label, (year_start, year_end) in WINDOWS:
        for experiment in SCENARIO_ORDER:
            future_rates = aggregate_period_rates(df, experiment, year_start, year_end)

            for group_name in GROUP_ORDER:
                changes = compute_group_percent_changes(
                    historical_rates, future_rates, group_name, GROUP_BASINS[group_name]
                )
                mean_change, sem_change = summarize_series(changes)
                rows.append(
                    {
                        "panel": panel_label,
                        "window_label": window_label,
                        "group": group_name,
                        "experiment": experiment,
                        "mean_percent_change": mean_change,
                        "sem_percent_change": sem_change,
                    }
                )

    _, _, (late_start, late_end) = WINDOWS[-1]
    late_ssp245 = aggregate_period_rates(df, "ssp245", late_start, late_end)
    late_ssp585 = aggregate_period_rates(df, "ssp585", late_start, late_end)
    diff_panel_label, diff_window_label = DIFF_PANEL

    for group_name in GROUP_ORDER:
        ssp245_changes = compute_group_percent_changes(
            historical_rates, late_ssp245, group_name, GROUP_BASINS[group_name]
        )
        ssp585_changes = compute_group_percent_changes(
            historical_rates, late_ssp585, group_name, GROUP_BASINS[group_name]
        )
        diff_changes = ssp585_changes - ssp245_changes
        mean_change, sem_change = summarize_series(diff_changes)
        rows.append(
            {
                "panel": diff_panel_label,
                "window_label": diff_window_label,
                "group": group_name,
                "experiment": DIFF_EXPERIMENT,
                "mean_percent_change": mean_change,
                "sem_percent_change": sem_change,
            }
        )

    return pd.DataFrame(rows)


def plot_change_summary(summary: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.8), constrained_layout=True, sharey=True)
    axes = axes.ravel()
    x = np.arange(len(GROUP_ORDER))
    width = 0.36

    lower = summary["mean_percent_change"] - summary["sem_percent_change"].fillna(0.0)
    upper = summary["mean_percent_change"] + summary["sem_percent_change"].fillna(0.0)
    ymin = min(float(lower.min()), 0.0)
    ymax = max(float(upper.max()), 0.0)
    pad = max((ymax - ymin) * 0.15, 2.0)

    panel_specs = [(panel_label, window_label) for panel_label, window_label, _ in WINDOWS]
    panel_specs.append(DIFF_PANEL)

    for idx, (panel_label, window_label) in enumerate(panel_specs):
        ax = axes[idx]
        panel = summary[summary["panel"] == panel_label]

        experiments = [DIFF_EXPERIMENT] if panel_label == DIFF_PANEL[0] else SCENARIO_ORDER

        for exp_idx, experiment in enumerate(experiments):
            subset = panel[panel["experiment"] == experiment].set_index("group").reindex(GROUP_ORDER)
            offset = 0.0 if len(experiments) == 1 else (exp_idx - (len(experiments) - 1) / 2) * width
            ax.bar(
                x + offset,
                subset["mean_percent_change"],
                width=width,
                yerr=subset["sem_percent_change"],
                capsize=CAPSIZE,
                label=EXPERIMENT_LABELS.get(experiment, experiment),
                color=DIFF_COLOR if experiment == DIFF_EXPERIMENT else EXPERIMENT_COLORS.get(experiment, "#999999"),
                edgecolor="black",
                linewidth=0.8,
            )

        ax.text(
            -0.12,
            1.08,
            panel_label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=12,
            fontweight="bold",
        )
        ax.set_title(window_label, fontsize=12, pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels(GROUP_ORDER)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_ylim(ymin - pad, ymax + pad)

        if idx % 2 == 0:
            ax.set_ylabel("Change of tropical cyclone frequency (%)")
        if idx == 0:
            ax.legend(
                frameon=False,
                ncol=len(SCENARIO_ORDER),
                loc="upper left",
                bbox_to_anchor=(0.0, 1.02),
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def open_output(path: Path) -> None:
    resolved = path.resolve()
    try:
        if sys.platform == "darwin":
            subprocess.run(["open", str(resolved)], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        elif sys.platform.startswith("linux"):
            subprocess.run(["xdg-open", str(resolved)], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        elif sys.platform == "win32":
            os.startfile(str(resolved))
    except Exception:
        pass


def main() -> int:
    args = parse_args()
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)
    df = load_rate_table(args.input)
    summary = compute_change_summary(df)
    plot_change_summary(summary, args.output)
    open_output(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
