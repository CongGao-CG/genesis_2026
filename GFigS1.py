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
import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import gaussian_kde


DEFAULT_INPUT_DIR = Path("gmt")
DEFAULT_OUTPUT = Path("GFigS1.pdf")
SCENARIO_ORDER = ["ssp245", "ssp585"]
SCENARIO_LABELS = {
    "ssp245": r"$\mathbf{a}$",
    "ssp585": r"$\mathbf{b}$",
}
SCENARIO_TITLES = {
    "ssp245": "SSP245",
    "ssp585": "SSP585",
}
SCENARIO_COLORS = {
    "ssp245": "#E69F00",
    "ssp585": "#E45756",
}
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
            "Plot probability densities of late-century (2065-2099) global warming "
            "relative to present day (1980-2014) from the gmtdiff_Amon files."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Directory containing gmtdiff netCDF files (default: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output figure path (default: {DEFAULT_OUTPUT})",
    )
    return parser.parse_args()


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
    missing_experiments = set(SCENARIO_ORDER) - set(df["experiment"])
    if missing_experiments:
        raise ValueError(
            f"Missing gmtdiff files for experiments: {sorted(missing_experiments)}"
        )

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
        df[df["experiment"] == "ssp585"]
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


def evaluate_density(values: np.ndarray, x_grid: np.ndarray) -> np.ndarray:
    if len(values) == 1:
        bandwidth = 0.05
        return np.exp(-0.5 * ((x_grid - values[0]) / bandwidth) ** 2) / (
            bandwidth * np.sqrt(2.0 * np.pi)
        )

    try:
        kde = gaussian_kde(values)
        return kde(x_grid)
    except np.linalg.LinAlgError:
        std = float(np.std(values, ddof=1))
        bandwidth = max(std * 0.3, 0.05)
        kernels = np.exp(-0.5 * ((x_grid[:, None] - values[None, :]) / bandwidth) ** 2)
        kernels /= bandwidth * np.sqrt(2.0 * np.pi)
        return kernels.mean(axis=1)


def build_density_frame(df: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], np.ndarray]:
    frames: dict[str, pd.DataFrame] = {}
    values = df["mean_gmtdiff"].to_numpy()
    x_min = float(values.min())
    x_max = float(values.max())
    x_span = x_max - x_min
    pad = max(x_span * 0.18, 0.15)
    x_grid = np.linspace(x_min - pad, x_max + pad, 400)

    for experiment in SCENARIO_ORDER:
        subset = df[df["experiment"] == experiment].copy()
        subset["density"] = np.nan
        density = evaluate_density(subset["mean_gmtdiff"].to_numpy(), x_grid)
        frames[experiment] = pd.DataFrame({"x": x_grid, "density": density})

    return frames, x_grid


def select_label_mask(
    subset: pd.DataFrame,
    x_min: float,
    x_max: float,
    forced_numbers: set[int],
) -> np.ndarray:
    x_values = subset["mean_gmtdiff"].to_numpy()
    numbers = subset["model_number"].to_numpy()
    keep = np.zeros(len(subset), dtype=bool)
    min_spacing = max((x_max - x_min) * 0.055, 0.14)
    last_labeled_x = float("-inf")

    for idx, (x_value, number) in enumerate(zip(x_values, numbers)):
        force = int(number) in forced_numbers
        if force or x_value - last_labeled_x >= min_spacing:
            keep[idx] = True
            last_labeled_x = float(x_value)

    for idx, number in enumerate(numbers):
        if int(number) in forced_numbers:
            keep[idx] = True

    return keep


def add_model_number_annotations(
    ax: plt.Axes,
    subset: pd.DataFrame,
    experiment: str,
    x_min: float,
    x_max: float,
    tick_height: float,
    label_y: float,
    forced_numbers: set[int],
) -> None:
    keep_mask = select_label_mask(subset, x_min, x_max, forced_numbers)
    for (_, row), keep in zip(subset.iterrows(), keep_mask):
        if not keep:
            continue
        ax.text(
            float(row["mean_gmtdiff"]),
            label_y,
            f"{int(row['model_number'])}",
            ha="center",
            va="center",
            fontsize=7.3,
            bbox={
                "boxstyle": "square,pad=0.16" if experiment == "ssp245" else "circle,pad=0.18",
                "facecolor": "white",
                "edgecolor": "black",
                "linewidth": 0.8,
            },
            clip_on=False,
        )


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
        "fontsize": 9.0,
        "fontfamily": "monospace",
        "bbox": {
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.85,
            "pad": 3.5,
        },
    }
    ax.text(0.10, 0.93, left_text, ha="left", **text_style)
    ax.text(0.64, 0.93, right_text, ha="left", **text_style)


def plot_density_panels(df: pd.DataFrame, output_path: Path) -> None:
    df, model_key = assign_model_numbers(df)
    density_frames, x_grid = build_density_frame(df)
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(8.2, 7.6),
        sharex=True,
        constrained_layout=True,
    )
    fig.set_constrained_layout_pads(hspace=0.14, h_pad=0.08)
    fig.supylabel("Probability density")
    fig.supxlabel("Late-century global warming level relative to present day (°C)")

    global_density_max = max(
        float(density_frames[experiment]["density"].max()) for experiment in SCENARIO_ORDER
    )
    tick_height = global_density_max * 0.06
    label_y = global_density_max * 0.11
    forced_numbers = {1, int(model_key["model_number"].max())}

    for ax, experiment in zip(axes, SCENARIO_ORDER):
        subset = df[df["experiment"] == experiment].sort_values("mean_gmtdiff")
        density_df = density_frames[experiment]
        color = SCENARIO_COLORS[experiment]

        ax.fill_between(
            density_df["x"],
            density_df["density"],
            color=color,
            alpha=0.22,
            linewidth=0,
        )
        ax.plot(
            density_df["x"],
            density_df["density"],
            color=color,
            linewidth=2.0,
        )
        ax.vlines(
            subset["mean_gmtdiff"],
            0.0,
            tick_height,
            color="black",
            linewidth=1.0,
        )
        add_model_number_annotations(
            ax,
            subset,
            experiment,
            float(x_grid[0]),
            float(x_grid[-1]),
            tick_height,
            label_y,
            forced_numbers,
        )
        ax.set_title(SCENARIO_LABELS[experiment], loc="left", fontsize=11, pad=8)
        ax.set_title(SCENARIO_TITLES[experiment], loc="center", fontsize=11, pad=8)
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlim(x_grid[0], x_grid[-1])
        ax.set_ylim(0.0, global_density_max * 1.15)
        ax.tick_params(axis="x", labelsize=9)
        ax.tick_params(axis="y", labelsize=9)

    add_model_key(axes[1], model_key)

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
    df = load_gmtdiff_means(args.input_dir)
    plot_density_panels(df, args.output)
    open_output(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
