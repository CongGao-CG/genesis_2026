from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import subprocess
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(SCRIPT_DIR / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(SCRIPT_DIR / ".cache"))

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["axes.unicode_minus"] = True

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.transforms import ScaledTranslation
import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import linregress, t


DEFAULT_RATE_INPUT = Path("g1_tc_by_basin_model_experiment_year.tsv")
DEFAULT_GMT_INPUT_DIR = Path("gmt")
DEFAULT_OUTPUT = Path("GFig4.pdf")
BASIN_ORDER = ["AS", "BoB", "WNP", "ENP", "NA", "SI", "SP"]
BASELINE_EXPERIMENT = "historical"
BASELINE_YEARS = (1980, 2014)
SCENARIO_ORDER = ["ssp245", "ssp585"]
ORDERING_EXPERIMENT = "ssp585"
TARGET_YEARS = (2015, 2099)
POSITIVE_TREND_COLOR = "#C73E3A"
NEGATIVE_TREND_COLOR = "#4C78A8"
ZERO_TREND_COLOR = "#7F7F7F"
LATE_WINDOW_YEARS = (2065, 2099)
PRESENT_WINDOW_YEARS = (1980, 2014)
FILE_PATTERN = re.compile(
    r"^gmtdiff_Amon_(?P<model>.+)_(?P<experiment>ssp245|ssp585)_(?P<variant>.+)\.nc$"
)
HISTORICAL_PATTERN = re.compile(
    r"^gmtdiff_Amon_(?P<model>.+)_historical_(?P<variant>.+)\.nc$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot model-by-model Global SSP585 TC-frequency trend against late-century "
            "global warming relative to 1850–1900."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_RATE_INPUT,
        help=f"Input TSV file for TC rates (default: {DEFAULT_RATE_INPUT})",
    )
    parser.add_argument(
        "--gmt-input-dir",
        type=Path,
        default=DEFAULT_GMT_INPUT_DIR,
        help=f"Directory containing gmtdiff netCDF files (default: {DEFAULT_GMT_INPUT_DIR})",
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

    missing_experiments = {BASELINE_EXPERIMENT, *SCENARIO_ORDER} - set(df["experiment"])
    if missing_experiments:
        raise ValueError(
            f"Missing required experiments in {path}: {sorted(missing_experiments)}"
        )

    missing_basins = set(BASIN_ORDER) - set(df["basin"])
    if missing_basins:
        raise ValueError(f"Missing required basins in {path}: {sorted(missing_basins)}")

    return df


def compute_global_slopes(df: pd.DataFrame) -> pd.DataFrame:
    global_df = (
        df[df["basin"].isin(BASIN_ORDER)]
        .groupby(["model", "experiment", "year"], as_index=False)["rate_per_year_per_realization"]
        .sum()
    )
    baseline = (
        global_df[
            (global_df["experiment"] == BASELINE_EXPERIMENT)
            & global_df["year"].between(*BASELINE_YEARS)
        ]
        .groupby("model", as_index=False)["rate_per_year_per_realization"]
        .mean()
        .rename(columns={"rate_per_year_per_realization": "historical_mean_rate"})
    )

    zero_baseline = baseline[baseline["historical_mean_rate"] == 0]["model"].tolist()
    if zero_baseline:
        raise ValueError(
            "Historical Global rate is zero for models, so percent change is undefined: "
            f"{zero_baseline[:6]}"
        )

    change_df = global_df.merge(
        baseline,
        on="model",
        how="left",
        validate="many_to_one",
    )
    missing_baseline = change_df[change_df["historical_mean_rate"].isna()]["model"].unique()
    if len(missing_baseline) > 0:
        raise ValueError(
            "Missing historical Global rate for models: "
            f"{missing_baseline[:6].tolist()}"
        )

    change_df["change_percent"] = (
        change_df["rate_per_year_per_realization"] - change_df["historical_mean_rate"]
    ).div(change_df["historical_mean_rate"]).mul(100.0)

    rows: list[dict[str, float | str]] = []
    future_df = change_df[
        (change_df["experiment"].isin(SCENARIO_ORDER))
        & change_df["year"].between(*TARGET_YEARS)
    ]
    for (model, experiment), subset in future_df.groupby(["model", "experiment"]):
        subset = subset.sort_values("year")
        if len(subset) < 2:
            continue
        result = linregress(subset["year"], subset["change_percent"])
        rows.append(
            {
                "model": model,
                "experiment": experiment,
                "slope": float(result.slope),
            }
        )

    return pd.DataFrame(rows).sort_values(["experiment", "model"]).reset_index(drop=True)


def parse_file_metadata(path: Path) -> dict[str, str]:
    match = FILE_PATTERN.match(path.name)
    if match is None:
        raise ValueError(f"Unexpected gmtdiff filename: {path.name}")
    return match.groupdict()


def compute_period_mean(path: Path, years: tuple[int, int]) -> float:
    with xr.open_dataset(path) as ds:
        if "gmt" not in ds.data_vars:
            raise ValueError(f"Missing 'gmt' variable in {path}")
        da = ds["gmt"].squeeze(drop=True)
        if "time" not in da.coords:
            raise ValueError(f"Missing 'time' coordinate in {path}")
        year_mask = da["time"].dt.year.isin(np.arange(years[0], years[1] + 1))
        window = da.where(year_mask, drop=True)
        if window.sizes.get("time", 0) == 0:
            raise ValueError(f"No data in {path} for years {years[0]}-{years[1]}")
        return float(window.mean(skipna=True).item())


def load_present_day_means(input_dir: Path) -> dict[str, float]:
    present: dict[str, float] = {}
    for path in sorted(input_dir.glob("gmtdiff_Amon_*.nc")):
        match = HISTORICAL_PATTERN.match(path.name)
        if match is None:
            continue
        model = match.group("model")
        if model in present:
            raise ValueError(f"Duplicate historical gmtdiff files for model {model}")
        present[model] = compute_period_mean(path, PRESENT_WINDOW_YEARS)
    if not present:
        raise ValueError(f"No historical gmtdiff files found in {input_dir}")
    return present


def load_gmtdiff_means(input_dir: Path) -> pd.DataFrame:
    present = load_present_day_means(input_dir)

    rows: list[dict[str, object]] = []
    for path in sorted(input_dir.glob("gmtdiff_Amon_*.nc")):
        # Skip files for other experiments (e.g. the historical baselines) that share
        # the gmtdiff_Amon_ prefix but do not match the ssp245/ssp585 pattern.
        if FILE_PATTERN.match(path.name) is None:
            continue
        metadata = parse_file_metadata(path)
        model = metadata["model"]
        if model not in present:
            raise ValueError(f"No historical (present-day) gmtdiff file for model {model}")
        # mean_gmtdiff is late-century warming expressed relative to present day:
        # (2065-2099 mean) - (1980-2014 historical mean).
        late_century = compute_period_mean(path, LATE_WINDOW_YEARS)
        rows.append(
            {
                "model": model,
                "experiment": metadata["experiment"],
                "variant": metadata["variant"],
                "mean_gmtdiff": late_century - present[model],
            }
        )

    if not rows:
        raise ValueError(f"No gmtdiff_Amon ssp245/ssp585 files found in {input_dir}")

    df = pd.DataFrame(rows)
    duplicate_keys = ["model", "experiment"]
    duplicates = df[df.duplicated(duplicate_keys, keep=False)][duplicate_keys]
    if not duplicates.empty:
        entries = duplicates.drop_duplicates().to_dict("records")
        raise ValueError(
            f"Duplicate gmtdiff files found for model/experiment pairs: {entries[:6]}"
        )

    return df.sort_values(["experiment", "model"]).reset_index(drop=True)


def assign_model_numbers(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordering = (
        df[df["experiment"] == ORDERING_EXPERIMENT]
        .sort_values("mean_gmtdiff")
        .reset_index(drop=True)[["model", "mean_gmtdiff"]]
        .copy()
    )
    ordering["model_number"] = np.arange(1, len(ordering) + 1)
    numbered = df.merge(
        ordering[["model", "model_number"]],
        on="model",
        how="left",
        validate="many_to_one",
    )
    return numbered, ordering[["model", "model_number"]]


def build_scatter_table(rate_df: pd.DataFrame, gmt_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    slopes = compute_global_slopes(rate_df)
    numbered_gmt, model_key = assign_model_numbers(gmt_df)
    target_gmt = numbered_gmt[numbered_gmt["experiment"].isin(SCENARIO_ORDER)][
        ["model", "experiment", "mean_gmtdiff", "model_number"]
    ]
    scatter_df = target_gmt.merge(
        slopes,
        on=["model", "experiment"],
        how="inner",
        validate="one_to_one",
    ).sort_values(["experiment", "mean_gmtdiff"])

    target_pairs = set(target_gmt[["model", "experiment"]].itertuples(index=False, name=None))
    scatter_pairs = set(scatter_df[["model", "experiment"]].itertuples(index=False, name=None))
    slope_pairs = set(slopes[["model", "experiment"]].itertuples(index=False, name=None))
    missing_rate_models = sorted(target_pairs - scatter_pairs)
    missing_gmt_models = sorted(slope_pairs - scatter_pairs)
    if missing_rate_models or missing_gmt_models:
        raise ValueError(
            "Model/experiment mismatch between TC slopes and gmtdiff means: "
            f"missing_rate_models={missing_rate_models[:6]}, "
            f"missing_gmt_models={missing_gmt_models[:6]}"
        )

    return scatter_df.reset_index(drop=True), model_key


def add_model_key(ax: plt.Axes, model_key: pd.DataFrame) -> None:
    key_rows = model_key.sort_values("model_number").reset_index(drop=True)
    left_rows = key_rows.iloc[:7]
    right_rows = key_rows.iloc[7:]
    left_text = "\n".join(
        f"{int(row.model_number)}. {row.model}" for row in left_rows.itertuples()
    )
    right_text = "\n".join(
        f"{int(row.model_number)}. {row.model}" for row in right_rows.itertuples()
    )
    text_style = {
        "transform": ax.transAxes,
        "va": "top",
        "fontsize": 8.5,
        "fontfamily": "monospace",
        "bbox": {
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.85,
            "pad": 3.2,
        },
    }
    ax.text(0.08, 0.96, left_text, ha="left", **text_style)
    ax.text(0.60, 0.96, right_text, ha="left", **text_style)


def format_p_value(p_value: float) -> str:
    if np.isnan(p_value):
        return "p = NA"
    if p_value < 0.001:
        return "p < 0.001"
    return f"p = {round(p_value, 3):.3f}"


def format_regression_annotation(n_points: int, p_value: float) -> str:
    return f"n = {n_points}, {format_p_value(p_value)}"


def format_slope_value(slope: float) -> str:
    rounded_slope = round(slope, 4)
    if np.isnan(rounded_slope):
        return "slope = NA"
    if rounded_slope < 0:
        return f"slope = \N{MINUS SIGN}{abs(rounded_slope):.4f}"
    return f"slope = {rounded_slope:.4f}"


def format_regression_annotation(n_points: int, slope: float, p_value: float) -> str:
    return f"{format_slope_value(slope)}, {format_p_value(p_value)}"


def trend_color(slope: float) -> str:
    if slope > 0:
        return POSITIVE_TREND_COLOR
    if slope < 0:
        return NEGATIVE_TREND_COLOR
    return ZERO_TREND_COLOR


def compute_regression_band(
    scatter_df: pd.DataFrame, x_grid: np.ndarray, confidence_level: float = 0.95
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, int]:
    # Fit all points with the line forced through the origin (intercept = 0): y = slope * x.
    x = scatter_df["mean_gmtdiff"].to_numpy(dtype=float)
    y = scatter_df["slope"].to_numpy(dtype=float)
    n_points = int(len(x))
    sxx = float(np.sum(x**2))
    if n_points < 2 or sxx == 0:
        nan_values = np.full_like(x_grid, np.nan, dtype=float)
        return nan_values, nan_values, nan_values, float("nan"), float("nan"), n_points

    slope = float(np.sum(x * y) / sxx)
    y_fit = slope * x_grid

    dof = n_points - 1
    residuals = y - slope * x
    residual_std = np.sqrt(np.sum(residuals**2) / dof)
    se_slope = residual_std / np.sqrt(sxx)
    if se_slope == 0:
        return y_fit, y_fit, y_fit, slope, float("nan"), n_points

    t_stat = slope / se_slope
    p_value = float(2.0 * t.sf(abs(t_stat), dof))

    alpha = 1.0 - confidence_level
    t_crit = float(t.ppf(1.0 - alpha / 2.0, dof))
    # Std error of the mean response at x for a through-origin fit: residual_std * |x| / sqrt(Sxx).
    band = t_crit * residual_std * np.abs(x_grid) / np.sqrt(sxx)
    return y_fit, y_fit - band, y_fit + band, slope, p_value, n_points


def plot_scatter(scatter_df: pd.DataFrame, model_key: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(
        1,
        1,
        figsize=(8.4, 6.2),
        constrained_layout=True,
    )

    for experiment in SCENARIO_ORDER:
        subset = scatter_df[scatter_df["experiment"] == experiment]
        marker = "s" if experiment == "ssp245" else "o"
        ax.scatter(
            subset["mean_gmtdiff"],
            subset["slope"],
            s=42,
            c="white",
            edgecolor="none",
            marker=marker,
            zorder=2,
        )

        for row in subset.itertuples():
            ax.text(
                float(row.mean_gmtdiff),
                float(row.slope),
                f"{int(row.model_number)}",
                ha="center",
                va="center",
                fontsize=7.1,
                color="black",
                bbox={
                    "boxstyle": "square,pad=0.16" if experiment == "ssp245" else "circle,pad=0.18",
                    "facecolor": "white",
                    "edgecolor": "black",
                    "linewidth": 0.8,
                },
                zorder=3,
            )

    ax.axhline(0.0, color="black", linewidth=0.8, zorder=1)
    ax.set_xlabel("Late-century global warming level relative to present day (°C)")
    ax.set_ylabel(r"Global tropical cyclone frequency trend (% year$^{-1}$)")
    ax.grid(color="#D9D9D9", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", labelsize=9)
    ax.tick_params(axis="y", labelsize=9)

    x_min = float(scatter_df["mean_gmtdiff"].min())
    x_max = float(scatter_df["mean_gmtdiff"].max())
    y_min = float(scatter_df["slope"].min())
    y_max = float(scatter_df["slope"].max())
    x_pad = max((x_max - x_min) * 0.12, 0.12)
    y_pad = max((y_max - y_min) * 0.18, 0.06)
    x_limits = (0.0, x_max + x_pad)
    x_grid = np.linspace(*x_limits, 200)
    y_fit, y_lower, y_upper, slope, p_value, n_points = compute_regression_band(
        scatter_df, x_grid
    )

    if np.isfinite(y_fit).any():
        ax.fill_between(
            x_grid,
            y_lower,
            y_upper,
            color="#7F7F7F",
            alpha=0.15,
            linewidth=0,
            zorder=0,
        )
        ax.plot(
            x_grid,
            y_fit,
            color="black",
            linewidth=1.4,
            linestyle="--",
            zorder=1,
        )

    y_limits = (
        min(y_min - y_pad, float(np.nanmin(y_lower)) if np.isfinite(y_lower).any() else y_min),
        max(y_max + y_pad, float(np.nanmax(y_upper)) if np.isfinite(y_upper).any() else y_max),
    )
    ax.set_xlim(*x_limits)
    ax.set_ylim(*y_limits)
    ax.text(
        0.98,
        0.04,
        format_regression_annotation(n_points, slope, p_value),
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
    )
    # Scenario is encoded by marker shape (square = SSP245, circle = SSP585).
    scenario_handles = [
        Line2D(
            [0],
            [0],
            marker="s",
            linestyle="None",
            markerfacecolor="white",
            markeredgecolor="black",
            markeredgewidth=0.8,
            color="none",
            markersize=7,
            label="SSP245",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor="white",
            markeredgecolor="black",
            markeredgewidth=0.8,
            color="none",
            markersize=7,
            label="SSP585",
        ),
    ]
    ax.legend(
        handles=scenario_handles,
        loc="upper right",
        frameon=False,
        fontsize=9,
        handletextpad=0.6,
    )

    # Model-number key intentionally omitted; call add_model_key(ax, model_key) to restore it.

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
    rate_df = load_rate_table(args.input)
    gmt_df = load_gmtdiff_means(args.gmt_input_dir)
    scatter_df, model_key = build_scatter_table(rate_df, gmt_df)
    plot_scatter(scatter_df, model_key, args.output)
    open_output(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
