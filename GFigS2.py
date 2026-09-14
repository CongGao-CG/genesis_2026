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

from matplotlib import colors, transforms
import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_INPUT = Path("g1_tc_by_basin_model_experiment_year.tsv")
DEFAULT_OUTPUT = Path("GFigS2.pdf")
BASIN_ORDER = ["AS", "BoB", "WNP", "ENP", "NA", "SI", "SP"]
GROUP_BASINS = {
    "Global": BASIN_ORDER,
    "NH": ["AS", "BoB", "WNP", "ENP", "NA"],
    "SH": ["SI", "SP"],
}
GROUP_ORDER = ["Global", "NH", "SH"]
BASELINE_EXPERIMENT = "historical"
BASELINE_YEARS = (1980, 2014)
PANEL_SPECS = [
    ("a", "Near-term (2015–2049) SSP245", "ssp245", (2015, 2049)),
    ("b", "Near-term (2015–2049) SSP585", "ssp585", (2015, 2049)),
    ("c", "Mid-century (2040–2074) SSP245", "ssp245", (2040, 2074)),
    ("d", "Mid-century (2040–2074) SSP585", "ssp585", (2040, 2074)),
    ("e", "Late-century (2065–2099) SSP245", "ssp245", (2065, 2099)),
    ("f", "Late-century (2065–2099) SSP585", "ssp585", (2065, 2099)),
]
DIFF_PANEL = ("g", "Late-century SSP585 − SSP245")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot Global/NH/SH percent changes in TC frequency for near-term, "
            "mid-century, and late-century windows, plus the late-century "
            "SSP585 minus SSP245 difference, from "
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

    missing_basins = set(BASIN_ORDER) - set(df["basin"])
    if missing_basins:
        raise ValueError(f"Missing required basins in {path}: {sorted(missing_basins)}")

    missing_experiments = {BASELINE_EXPERIMENT, "ssp245", "ssp585"} - set(df["experiment"])
    if missing_experiments:
        raise ValueError(
            f"Missing required experiments in {path}: {sorted(missing_experiments)}"
        )

    return df


def ordered_models(df: pd.DataFrame) -> list[str]:
    return df["model"].drop_duplicates().tolist()


def aggregate_group_period_rates(
    df: pd.DataFrame, experiment: str, year_start: int, year_end: int
) -> pd.Series:
    models = ordered_models(df)
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

    rows: list[dict[str, object]] = []
    for group in GROUP_ORDER:
        group_series = series.loc[GROUP_BASINS[group]].groupby(level=["model", "year"]).sum()
        mean_rates = group_series.groupby(level="model").mean()
        for model, rate in mean_rates.items():
            rows.append(
                {
                    "group": group,
                    "model": model,
                    "rate_per_year_per_realization": rate,
                }
            )

    return pd.DataFrame(rows).set_index(["group", "model"])["rate_per_year_per_realization"]


def compute_percent_change_matrix(
    historical_rates: pd.Series,
    future_rates: pd.Series,
    models: list[str],
    panel_label: str,
    title: str,
) -> pd.DataFrame:
    zero_baseline = historical_rates.index[historical_rates == 0].tolist()
    if zero_baseline:
        raise ValueError(
            "Historical mean rate is zero for group/model pairs, so percent change is "
            f"undefined: {format_entries(zero_baseline)}"
        )

    percent_change = (
        (future_rates - historical_rates).div(historical_rates).mul(100.0)
    )
    zero_changes = percent_change[percent_change == 0].index.tolist()
    if zero_changes:
        raise ValueError(
            f"Exact zero percent change found in panel {panel_label} ({title}): "
            f"{format_entries(zero_changes)}"
        )
    return percent_change.unstack("model").reindex(index=GROUP_ORDER, columns=models)


def row_sign_annotations(change: pd.DataFrame) -> list[tuple[int, str, str]]:
    annotations: list[tuple[int, str, str]] = []
    for row_idx, row_label in enumerate(change.index):
        row = change.loc[row_label]
        positive_count = int((row > 0).sum())
        negative_count = int((row < 0).sum())

        if positive_count >= 13:
            annotations.append((row_idx, "**", "red"))
        elif positive_count >= 10:
            annotations.append((row_idx, "*", "red"))
        elif negative_count >= 13:
            annotations.append((row_idx, "**", "blue"))
        elif negative_count >= 10:
            annotations.append((row_idx, "*", "blue"))

    return annotations


def validate_no_exact_zeros(change: pd.DataFrame, panel_label: str, title: str) -> None:
    zero_mask = change == 0
    if zero_mask.to_numpy().any():
        zero_entries = [
            (row_label, col_label)
            for row_label in change.index
            for col_label in change.columns
            if zero_mask.loc[row_label, col_label]
        ]
        raise ValueError(
            f"Exact zero change found in panel {panel_label} ({title}): "
            f"{format_entries(zero_entries)}"
        )


def build_panel_matrices(df: pd.DataFrame) -> list[tuple[str, str, pd.DataFrame]]:
    models = ordered_models(df)
    historical_rates = aggregate_group_period_rates(df, BASELINE_EXPERIMENT, *BASELINE_YEARS)

    changes: list[tuple[str, str, pd.DataFrame]] = []
    late_century_matrices: dict[str, pd.DataFrame] = {}
    for panel_label, title, scenario, (year_start, year_end) in PANEL_SPECS:
        future_rates = aggregate_group_period_rates(df, scenario, year_start, year_end)
        matrix = compute_percent_change_matrix(
            historical_rates, future_rates, models, panel_label, title
        )
        changes.append((panel_label, title, matrix))
        if panel_label in {"e", "f"}:
            late_century_matrices[panel_label] = matrix

    diff_label, diff_title = DIFF_PANEL
    diff_matrix = late_century_matrices["f"] - late_century_matrices["e"]
    validate_no_exact_zeros(diff_matrix, diff_label, diff_title)
    changes.append((diff_label, diff_title, diff_matrix))
    return changes


def plot_change_heatmaps(changes: list[tuple[str, str, pd.DataFrame]], output_path: Path) -> None:
    vmax = max(change.abs().max().max() for _, _, change in changes)
    if pd.isna(vmax) or vmax == 0:
        vmax = 1.0
    norm = colors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    fig, axes = plt.subplots(
        nrows=4,
        ncols=2,
        figsize=(18, 12.5),
        constrained_layout=True,
    )
    flat_axes = axes.ravel().tolist()
    plot_axes = flat_axes[: len(changes)]
    flat_axes[-1].axis("off")

    image = None
    for ax, (panel_label, title, change) in zip(plot_axes, changes):
        image = ax.imshow(change, cmap="RdBu_r", norm=norm, aspect="auto")

        ax.set_title(title, fontsize=12, pad=10)
        ax.text(
            0.0,
            1.04,
            panel_label,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontweight="bold",
            fontsize=14,
            clip_on=False,
        )

        ax.set_xticks(range(len(change.columns)))
        ax.set_xticklabels(change.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(change.index)))
        ax.set_yticklabels(change.index)

        ax.set_xticks([x - 0.5 for x in range(1, len(change.columns))], minor=True)
        ax.set_yticks([y - 0.5 for y in range(1, len(change.index))], minor=True)
        ax.grid(which="minor", color="white", linewidth=1.0)
        ax.tick_params(which="minor", bottom=False, left=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        marker_transform = transforms.blended_transform_factory(ax.transAxes, ax.transData)
        for row_idx, marker, color in row_sign_annotations(change):
            ax.text(
                1.02,
                row_idx,
                marker,
                transform=marker_transform,
                ha="left",
                va="center",
                color=color,
                fontweight="bold",
                fontsize=12,
                clip_on=False,
            )

    cbar = fig.colorbar(image, ax=plot_axes, pad=0.04, shrink=0.96)
    cbar.set_label("Change of tropical cyclone frequency (%)")

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
    changes = build_panel_matrices(df)
    plot_change_heatmaps(changes, args.output)
    open_output(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
