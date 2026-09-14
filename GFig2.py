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
matplotlib.rcParams["axes.unicode_minus"] = True

from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import linregress


DEFAULT_INPUT = Path("g1_tc_by_basin_model_experiment_year.tsv")
DEFAULT_OUTPUT = Path("GFig2.pdf")
BASIN_ORDER = ["AS", "BoB", "WNP", "ENP", "NA", "SI", "SP"]
GROUP_ORDER = ["Global", "NH", "SH"]
GROUP_BASINS = {
    "Global": BASIN_ORDER,
    "NH": ["AS", "BoB", "WNP", "ENP", "NA"],
    "SH": ["SI", "SP"],
}
BASELINE_EXPERIMENT = "historical"
BASELINE_YEARS = (1980, 2014)
EXPERIMENT_ORDER = ["historical", "ssp245", "ssp585"]
EXPERIMENT_LABELS = {
    "historical": "Historical (1980–2014)",
    "ssp245": "SSP245 (2015–2099)",
    "ssp585": "SSP585 (2015–2099)",
}
EXPERIMENT_COLORS = {
    "historical": "#4C78A8",
    "ssp245": "#E69F00",
    "ssp585": "#E45756",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot yearly changes in grouped TC genesis rates relative to the "
            "historical mean from g1_tc_by_basin_model_experiment_year.tsv."
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

    missing_experiments = set(EXPERIMENT_ORDER) - set(df["experiment"])
    if missing_experiments:
        raise ValueError(
            f"Missing required experiments in {path}: {sorted(missing_experiments)}"
        )

    missing_basins = set(BASIN_ORDER) - set(df["basin"])
    if missing_basins:
        raise ValueError(f"Missing required basins in {path}: {sorted(missing_basins)}")

    return df


def build_group_rate_table(df: pd.DataFrame) -> pd.DataFrame:
    group_tables = []
    for group, basins in GROUP_BASINS.items():
        group_df = (
            df[df["basin"].isin(basins)]
            .groupby(["model", "experiment", "year"], as_index=False)["rate_per_year_per_realization"]
            .sum()
        )
        group_df.insert(0, "group", group)
        group_tables.append(group_df)
    return pd.concat(group_tables, ignore_index=True)


def compute_historical_baseline(group_df: pd.DataFrame) -> pd.DataFrame:
    baseline = (
        group_df[
            (group_df["experiment"] == BASELINE_EXPERIMENT)
            & group_df["year"].between(*BASELINE_YEARS)
        ]
        .groupby(["group", "model"], as_index=False)["rate_per_year_per_realization"]
        .mean()
        .rename(columns={"rate_per_year_per_realization": "historical_mean_rate"})
    )

    zero_baseline = baseline[baseline["historical_mean_rate"] == 0][["group", "model"]]
    if not zero_baseline.empty:
        raise ValueError(
            "Historical mean rate is zero for group/model pairs, so percent change is "
            f"undefined: {zero_baseline.to_dict('records')[:6]}"
        )

    return baseline


def summarize_time_series(df: pd.DataFrame) -> pd.DataFrame:
    group_df = build_group_rate_table(df)
    baseline = compute_historical_baseline(group_df)
    change_df = group_df.merge(
        baseline,
        on=["group", "model"],
        how="left",
        validate="many_to_one",
    )

    missing_baseline = change_df[change_df["historical_mean_rate"].isna()][
        ["group", "model"]
    ].drop_duplicates()
    if not missing_baseline.empty:
        raise ValueError(
            "Missing historical mean rate for group/model pairs: "
            f"{missing_baseline.to_dict('records')[:6]}"
        )

    change_df["change_percent"] = (
        change_df["rate_per_year_per_realization"] - change_df["historical_mean_rate"]
    ).div(change_df["historical_mean_rate"]).mul(100.0)

    summary = (
        change_df.groupby(["group", "experiment", "year"], as_index=False)["change_percent"]
        .agg(mean_change="mean", sd_change="std", n_models="count")
    )
    summary["sd_change"] = summary["sd_change"].fillna(0.0)
    summary["se_change"] = summary["sd_change"] / summary["n_models"].pow(0.5)
    return summary


def compute_trend_stats(subset: pd.DataFrame) -> tuple[float, float] | None:
    trend_subset = subset[subset["year"].between(2015, 2099)].sort_values("year")
    if len(trend_subset) < 2:
        return None
    result = linregress(trend_subset["year"], trend_subset["mean_change"])
    return float(result.slope), float(result.pvalue)


def format_p_value(p_value: float) -> str:
    if p_value < 0.001:
        return "p < 0.001"
    return f"p = {round(p_value, 3):.3f}"


def format_slope(slope: float) -> str:
    rounded_slope = round(slope, 4)
    if rounded_slope < 0:
        return f"slope = \N{MINUS SIGN}{abs(rounded_slope):.4f}"
    return f"slope = {rounded_slope:.4f}"


def plot_group_time_series(summary: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(12.5, 8.5),
        sharex=True,
        constrained_layout=True,
    )
    fig.supylabel("Change of tropical cyclone frequency (%)")
    axes_flat = axes.ravel()

    for panel_index, (ax, group) in enumerate(zip(axes_flat, GROUP_ORDER)):
        group_summary = summary[summary["group"] == group]

        ymin = float("inf")
        ymax = float("-inf")

        for experiment in EXPERIMENT_ORDER:
            subset = (
                group_summary[group_summary["experiment"] == experiment]
                .sort_values("year")
                .reset_index(drop=True)
            )
            if subset.empty:
                continue

            years = subset["year"]
            mean_change = subset["mean_change"]
            se_change = subset["se_change"]
            lower = mean_change - se_change
            upper = mean_change + se_change

            ax.fill_between(
                years,
                lower,
                upper,
                color=EXPERIMENT_COLORS[experiment],
                alpha=0.18,
                linewidth=0,
            )
            ax.plot(
                years,
                mean_change,
                color=EXPERIMENT_COLORS[experiment],
                linewidth=2.0,
            )

            ymin = min(ymin, lower.min())
            ymax = max(ymax, upper.max())

        historical_2014 = group_summary[
            (group_summary["experiment"] == "historical") & (group_summary["year"] == 2014)
        ]
        if not historical_2014.empty:
            historical_change_2014 = historical_2014["mean_change"].iloc[0]
            for experiment in ("ssp245", "ssp585"):
                future_2015 = group_summary[
                    (group_summary["experiment"] == experiment) & (group_summary["year"] == 2015)
                ]
                if future_2015.empty:
                    continue
                ax.plot(
                    [2014, 2015],
                    [historical_change_2014, future_2015["mean_change"].iloc[0]],
                    color=EXPERIMENT_COLORS[experiment],
                    linewidth=2.0,
                )

        margin = (ymax - ymin) * 0.12
        if margin == 0:
            margin = max(max(abs(ymin), abs(ymax)) * 0.12, 1.0)

        ax.axvline(2014.5, color="#7F7F7F", linewidth=1.0, linestyle="--")
        ax.set_title(group, fontsize=11, pad=6)
        ax.set_ylim(ymin - margin, ymax + margin)
        ax.text(
            -0.14,
            1.04,
            chr(ord("a") + panel_index),
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold",
            ha="left",
            va="bottom",
        )
        for text_y, experiment in zip((0.98, 0.90), ("ssp245", "ssp585")):
            subset = group_summary[group_summary["experiment"] == experiment]
            trend_stats = compute_trend_stats(subset)
            if trend_stats is None:
                continue
            slope, p_value = trend_stats
            ax.text(
                0.98,
                text_y,
                f"{format_slope(slope)}, {format_p_value(p_value)}",
                transform=ax.transAxes,
                fontsize=8.5,
                color=EXPERIMENT_COLORS[experiment],
                ha="right",
                va="top",
            )
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    legend_ax = axes_flat[len(GROUP_ORDER)]
    legend_ax.axis("off")
    handles = [
        Line2D(
            [0],
            [0],
            color=EXPERIMENT_COLORS[experiment],
            linewidth=2.0,
            label=EXPERIMENT_LABELS[experiment],
        )
        for experiment in EXPERIMENT_ORDER
    ]
    divider_handle = Line2D(
        [0],
        [0],
        color="#7F7F7F",
        linewidth=1.0,
        linestyle="--",
        label="2014/2015 split",
    )
    legend_ax.legend(
        handles=handles + [divider_handle],
        loc="center",
        frameon=False,
        fontsize=10,
    )

    tick_years = [1980, 2000, 2015, 2050, 2099]
    for ax in axes_flat[: len(GROUP_ORDER)]:
        ax.set_xticks(tick_years)
        ax.tick_params(axis="x", labelbottom=True)
        ax.tick_params(axis="x", labelsize=9)
        ax.tick_params(axis="y", labelsize=9)

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
    summary = summarize_time_series(df)
    plot_group_time_series(summary, args.output)
    open_output(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
