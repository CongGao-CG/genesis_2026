from __future__ import annotations

import argparse
import itertools
import os
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(SCRIPT_DIR / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(SCRIPT_DIR / ".cache"))

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["axes.unicode_minus"] = True

import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerBase
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import statsmodels.api as sm

from matplotlib.text import Text


DEFAULT_INPUT = Path("g1_tc_by_basin_model_experiment_year.tsv")


DEFAULT_ONERETAINED_INPUT = Path("g1_oneretained_tc_by_basin_model_experiment_year.tsv")


DEFAULT_TWORETAINED_INPUT = Path("g1_tworetained_tc_by_basin_model_experiment_year.tsv")


DEFAULT_THREERETAINED_INPUT = Path("g1_threeretained_tc_by_basin_model_experiment_year.tsv")


BASIN_ORDER = ["AS", "BoB", "WNP", "ENP", "NA", "SI", "SP"]


GROUP_ORDER = ["Global", "NH", "SH"]


GROUP_BASINS = {
    "Global": BASIN_ORDER,
    "NH": ["AS", "BoB", "WNP", "ENP", "NA"],
    "SH": ["SI", "SP"],
}


BASELINE_EXPERIMENT = "historical"


BASELINE_YEARS = (1980, 2014)


FUTURE_EXPERIMENTS = ["ssp245", "ssp585"]


RETAINED_PREDICTOR_ORDER = ["av850", "shr", "rh600", "pi"]


RETAINED_PREDICTOR_LABELS = {
    "av850": "AV850",
    "shr": "SHR",
    "rh600": "RH600",
    "pi": "PI",
}


PREDICTOR_INDEX = {predictor: i for i, predictor in enumerate(RETAINED_PREDICTOR_ORDER)}


TWORETAINED_ORDER = [
    "av850_shr",
    "av850_rh600",
    "av850_pi",
    "shr_rh600",
    "shr_pi",
    "rh600_pi",
]


THREERETAINED_ORDER = [
    "av850_shr_rh600",
    "av850_shr_pi",
    "av850_rh600_pi",
    "shr_rh600_pi",
]


RETAINED_PREDICTOR_FILL_LETTERS = {
    "av850": "A",
    "shr": "S",
    "rh600": "R",
    "pi": "P",
}


PREDICTOR_BAR_FACE_COLOR = "#FFFFFF"


class LetterFillLegendItem:
    def __init__(self, letter: str) -> None:
        self.letter = letter


class LetterFillLegendHandler(HandlerBase):
    def create_artists(
        self,
        legend,
        orig_handle,
        xdescent,
        ydescent,
        width,
        height,
        fontsize,
        trans,
    ):
        x0 = -xdescent
        y0 = -ydescent
        rect = Rectangle(
            (x0, y0),
            width,
            height,
            facecolor=PREDICTOR_BAR_FACE_COLOR,
            edgecolor="black",
            linewidth=0.8,
            transform=trans,
        )
        artists = [rect]
        artists.append(
            Text(
                x=x0 + width * 0.5,
                y=y0 + height * 0.5,
                text=orig_handle.letter,
                ha="center",
                va="center",
                fontsize=fontsize * 0.8,
                color="#666666",
                transform=trans,
            )
        )
        return artists


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

    missing_experiments = set(["historical", *FUTURE_EXPERIMENTS]) - set(df["experiment"])
    if missing_experiments:
        raise ValueError(
            f"Missing required experiments in {path}: {sorted(missing_experiments)}"
        )

    missing_basins = set(BASIN_ORDER) - set(df["basin"])
    if missing_basins:
        raise ValueError(f"Missing required basins in {path}: {sorted(missing_basins)}")

    return df


def load_predictor_rate_table(
    path: Path, predictor_column: str, expected_values: list[str]
) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", keep_default_na=False)
    required = {
        "basin",
        "model",
        "experiment",
        "variant",
        predictor_column,
        "year",
        "rate_per_year_per_realization",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {path}: {sorted(missing)}")

    missing_experiments = set(FUTURE_EXPERIMENTS) - set(df["experiment"])
    if missing_experiments:
        raise ValueError(
            f"Missing required experiments in {path}: {sorted(missing_experiments)}"
        )

    missing_predictors = set(expected_values) - set(df[predictor_column])
    if missing_predictors:
        raise ValueError(
            f"Missing required predictors in {path}: {sorted(missing_predictors)}"
        )

    missing_basins = set(BASIN_ORDER) - set(df["basin"])
    if missing_basins:
        raise ValueError(f"Missing required basins in {path}: {sorted(missing_basins)}")

    return df


def load_oneretained_rate_table(path: Path) -> pd.DataFrame:
    return load_predictor_rate_table(path, "retained_predictor", RETAINED_PREDICTOR_ORDER)


def load_tworetained_rate_table(path: Path) -> pd.DataFrame:
    return load_predictor_rate_table(
        path, "retained_predictor_pair", TWORETAINED_ORDER
    )


def load_threeretained_rate_table(path: Path) -> pd.DataFrame:
    return load_predictor_rate_table(
        path, "retained_predictor_triple", THREERETAINED_ORDER
    )


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


def build_group_predictor_rate_table(
    df: pd.DataFrame, predictor_column: str
) -> pd.DataFrame:
    group_tables = []
    for group, basins in GROUP_BASINS.items():
        group_df = (
            df[df["basin"].isin(basins)]
            .groupby(
                ["model", "experiment", predictor_column, "year"],
                as_index=False,
            )["rate_per_year_per_realization"]
            .sum()
        )
        group_df.insert(0, "group", group)
        group_tables.append(group_df)
    return pd.concat(group_tables, ignore_index=True)


def build_group_oneretained_rate_table(df: pd.DataFrame) -> pd.DataFrame:
    return build_group_predictor_rate_table(df, "retained_predictor")


def build_group_tworetained_rate_table(df: pd.DataFrame) -> pd.DataFrame:
    return build_group_predictor_rate_table(df, "retained_predictor_pair")


def build_group_threeretained_rate_table(df: pd.DataFrame) -> pd.DataFrame:
    return build_group_predictor_rate_table(df, "retained_predictor_triple")


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


def apply_historical_baseline(
    group_df: pd.DataFrame, baseline: pd.DataFrame
) -> pd.DataFrame:
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
    return change_df


def compute_oneretained_change_table(
    oneretained_df: pd.DataFrame, baseline: pd.DataFrame
) -> pd.DataFrame:
    group_df = build_group_oneretained_rate_table(oneretained_df)
    return apply_historical_baseline(group_df, baseline)


def compute_tworetained_change_table(
    tworetained_df: pd.DataFrame, baseline: pd.DataFrame
) -> pd.DataFrame:
    group_df = build_group_tworetained_rate_table(tworetained_df)
    return apply_historical_baseline(group_df, baseline)


def compute_threeretained_change_table(
    threeretained_df: pd.DataFrame, baseline: pd.DataFrame
) -> pd.DataFrame:
    group_df = build_group_threeretained_rate_table(threeretained_df)
    return apply_historical_baseline(group_df, baseline)


def summarize_mean_change(change_df: pd.DataFrame) -> pd.DataFrame:
    return (
        change_df.groupby(["group", "experiment", "year"], as_index=False)["change_percent"]
        .mean()
        .rename(columns={"change_percent": "mean_change"})
    )


def summarize_predictor_mean_change(
    change_df: pd.DataFrame, predictor_column: str
) -> pd.DataFrame:
    return (
        change_df.groupby(
            ["group", predictor_column, "experiment", "year"], as_index=False
        )["change_percent"]
        .mean()
        .rename(columns={"change_percent": "mean_change"})
    )


def summarize_oneretained_mean_change(change_df: pd.DataFrame) -> pd.DataFrame:
    return summarize_predictor_mean_change(change_df, "retained_predictor")


def summarize_tworetained_mean_change(change_df: pd.DataFrame) -> pd.DataFrame:
    return summarize_predictor_mean_change(change_df, "retained_predictor_pair")


def summarize_threeretained_mean_change(change_df: pd.DataFrame) -> pd.DataFrame:
    return summarize_predictor_mean_change(change_df, "retained_predictor_triple")


def compute_method1_stats(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []

    for group in GROUP_ORDER:
        subset = summary[
            (summary["group"] == group)
            & (summary["experiment"].isin(FUTURE_EXPERIMENTS))
            & (summary["year"].between(2015, 2099))
        ].copy()
        if subset.empty:
            continue

        subset["year_centered"] = subset["year"] - 2015
        subset["is_ssp585"] = (subset["experiment"] == "ssp585").astype(int)
        subset["interaction"] = subset["year_centered"] * subset["is_ssp585"]

        design = sm.add_constant(subset[["year_centered", "is_ssp585", "interaction"]])
        fit = sm.OLS(subset["mean_change"], design).fit()

        rows.append({"group": group, "slope_diff": float(fit.params["interaction"])})

    return pd.DataFrame(rows)


def compute_predictor_method1_stats(
    summary: pd.DataFrame, predictor_column: str, key_order: list[str]
) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []

    for group in GROUP_ORDER:
        for predictor in key_order:
            subset = summary[
                (summary["group"] == group)
                & (summary[predictor_column] == predictor)
                & (summary["experiment"].isin(FUTURE_EXPERIMENTS))
                & (summary["year"].between(2015, 2099))
            ].copy()
            if subset.empty:
                continue

            subset["year_centered"] = subset["year"] - 2015
            subset["is_ssp585"] = (subset["experiment"] == "ssp585").astype(int)
            subset["interaction"] = subset["year_centered"] * subset["is_ssp585"]

            design = sm.add_constant(
                subset[["year_centered", "is_ssp585", "interaction"]]
            )
            fit = sm.OLS(subset["mean_change"], design).fit()

            rows.append(
                {
                    "group": group,
                    predictor_column: predictor,
                    "slope_diff": float(fit.params["interaction"]),
                }
            )

    return pd.DataFrame(rows)


def compute_oneretained_method1_stats(summary: pd.DataFrame) -> pd.DataFrame:
    return compute_predictor_method1_stats(
        summary, "retained_predictor", RETAINED_PREDICTOR_ORDER
    )


def compute_tworetained_method1_stats(summary: pd.DataFrame) -> pd.DataFrame:
    return compute_predictor_method1_stats(
        summary, "retained_predictor_pair", TWORETAINED_ORDER
    )


def compute_threeretained_method1_stats(summary: pd.DataFrame) -> pd.DataFrame:
    return compute_predictor_method1_stats(
        summary, "retained_predictor_triple", THREERETAINED_ORDER
    )


def ordered_combo_key(parts: tuple[str, ...] | list[str]) -> str:
    return "_".join(sorted(parts, key=PREDICTOR_INDEX.__getitem__))


def fill_text_for_combo(parts: tuple[str, ...] | list[str]) -> str:
    return " ".join(RETAINED_PREDICTOR_FILL_LETTERS[part] for part in parts)


def compute_twoway_interaction_stats(
    oneretained_method1_stats: pd.DataFrame,
    tworetained_method1_stats: pd.DataFrame,
) -> pd.DataFrame:
    single_lookup = oneretained_method1_stats.set_index(
        ["group", "retained_predictor"]
    )["slope_diff"]
    pair_lookup = tworetained_method1_stats.set_index(
        ["group", "retained_predictor_pair"]
    )["slope_diff"]
    rows: list[dict[str, float | str]] = []

    for group in GROUP_ORDER:
        for pair in TWORETAINED_ORDER:
            parts = tuple(pair.split("_"))
            rows.append(
                {
                    "group": group,
                    "effect_key": pair,
                    "effect_value": float(
                        pair_lookup.loc[(group, pair)]
                        - single_lookup.loc[(group, parts[0])]
                        - single_lookup.loc[(group, parts[1])]
                    ),
                    "fill_text": fill_text_for_combo(parts),
                    "effect_type": "two_way",
                }
            )

    return pd.DataFrame(rows)


def compute_threeway_interaction_stats(
    oneretained_method1_stats: pd.DataFrame,
    tworetained_method1_stats: pd.DataFrame,
    threeretained_method1_stats: pd.DataFrame,
) -> pd.DataFrame:
    single_lookup = oneretained_method1_stats.set_index(
        ["group", "retained_predictor"]
    )["slope_diff"]
    pair_lookup = tworetained_method1_stats.set_index(
        ["group", "retained_predictor_pair"]
    )["slope_diff"]
    triple_lookup = threeretained_method1_stats.set_index(
        ["group", "retained_predictor_triple"]
    )["slope_diff"]
    rows: list[dict[str, float | str]] = []

    for group in GROUP_ORDER:
        for triple in THREERETAINED_ORDER:
            parts = tuple(triple.split("_"))
            pair_keys = [
                ordered_combo_key((parts[0], parts[1])),
                ordered_combo_key((parts[0], parts[2])),
                ordered_combo_key((parts[1], parts[2])),
            ]
            rows.append(
                {
                    "group": group,
                    "effect_key": triple,
                    "effect_value": float(
                        triple_lookup.loc[(group, triple)]
                        - pair_lookup.loc[(group, pair_keys[0])]
                        - pair_lookup.loc[(group, pair_keys[1])]
                        - pair_lookup.loc[(group, pair_keys[2])]
                        + single_lookup.loc[(group, parts[0])]
                        + single_lookup.loc[(group, parts[1])]
                        + single_lookup.loc[(group, parts[2])]
                    ),
                    "fill_text": fill_text_for_combo(parts),
                    "effect_type": "three_way",
                }
            )

    return pd.DataFrame(rows)


def build_effect_offsets(effect_order: list[str]) -> tuple[np.ndarray, float]:
    count = len(effect_order)
    if count == 1:
        return np.array([0.0]), 0.28

    max_span = 0.78
    step = min(0.18, max_span / max(count - 1, 1))
    offsets = (np.arange(count, dtype=float) - (count - 1) / 2.0) * step
    bar_width = step * 0.82
    return offsets, bar_width


def add_group_separators(ax: plt.Axes) -> None:
    for xpos in [0.5, 1.5]:
        ax.axvline(xpos, color="#808080", linestyle="--", linewidth=0.9, zorder=0)


def add_centered_label_to_bar(ax: plt.Axes, bar, fill_text: str) -> None:
    if not fill_text:
        return
    height = float(bar.get_height())
    if pd.isna(height) or height == 0.0:
        return

    x = bar.get_x() + bar.get_width() * 0.5
    y = height * 0.5
    txt = ax.text(
        x,
        y,
        fill_text,
        ha="center",
        va="center",
        fontsize=7.0,
        color="#666666",
        zorder=bar.get_zorder() + 0.2,
        clip_on=True,
    )
    txt.set_clip_path(bar)


DEFAULT_OUTPUT = Path("GFig3.pdf")
MAIN_EFFECT_KEY = "total_trend_difference"
MAIN_EFFECT_LABEL = "Total trend difference (SSP585 − SSP245)"
MAIN_BAR_COLOR = "#BFBFBF"
INTERACTION_EFFECT_KEY = "interaction_contribution"
INTERACTION_EFFECT_LABEL = "Interaction contribution"
INTERACTION_EFFECT_HATCH = "x"
PANEL_TITLE = (
    "Total trend difference (SSP585 − SSP245), individual predictor contribution, "
    "and interaction contribution"
)


class CenteredHatchLegendItem:
    def __init__(self, hatch: str) -> None:
        self.hatch = hatch


class CenteredHatchLegendHandler(HandlerBase):
    def create_artists(
        self,
        legend,
        orig_handle,
        xdescent,
        ydescent,
        width,
        height,
        fontsize,
        trans,
    ):
        x0 = -xdescent
        y0 = -ydescent
        rect_width = width * 0.82
        rect_height = height * 0.52
        rect_x = x0 + (width - rect_width) * 0.5
        rect_y = y0 + (height - rect_height) * 0.5
        rect = Rectangle(
            (rect_x, rect_y),
            rect_width,
            rect_height,
            facecolor="white",
            edgecolor="black",
            linewidth=0.8,
            transform=trans,
        )
        artists = [rect]

        x_left = rect_x + rect_width * 0.24
        x_center = rect_x + rect_width * 0.5
        x_right = rect_x + rect_width * 0.76
        y_bottom = rect_y + rect_height * 0.24
        y_center = rect_y + rect_height * 0.5
        y_top = rect_y + rect_height * 0.76

        if orig_handle.hatch == "/":
            artists.append(
                Line2D(
                    [x_left, x_right],
                    [y_bottom, y_top],
                    color="black",
                    linewidth=1.0,
                    transform=trans,
                )
            )
        elif orig_handle.hatch == "x":
            artists.extend(
                [
                    Line2D(
                        [x_left, x_right],
                        [y_bottom, y_top],
                        color="black",
                        linewidth=1.0,
                        transform=trans,
                    ),
                    Line2D(
                        [x_left, x_right],
                        [y_top, y_bottom],
                        color="black",
                        linewidth=1.0,
                        transform=trans,
                    ),
                ]
            )
        elif orig_handle.hatch == ".":
            artists.append(
                Line2D(
                    [x_center],
                    [y_center],
                    marker="o",
                    markersize=3.2,
                    linestyle="None",
                    markeredgecolor="black",
                    markerfacecolor="none",
                    transform=trans,
                )
            )

        return artists


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot grouped SSP585 minus SSP245 differences in TC genesis-rate "
            "change with the total trend difference, individual predictor "
            "contributions, and summed interaction contribution."
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
    parser.add_argument(
        "--oneretained-input",
        type=Path,
        default=DEFAULT_ONERETAINED_INPUT,
        help=(
            "One-retained-predictor input TSV file "
            f"(default: {DEFAULT_ONERETAINED_INPUT})"
        ),
    )
    parser.add_argument(
        "--tworetained-input",
        type=Path,
        default=DEFAULT_TWORETAINED_INPUT,
        help=(
            "Two-retained-predictor input TSV file "
            f"(default: {DEFAULT_TWORETAINED_INPUT})"
        ),
    )
    parser.add_argument(
        "--threeretained-input",
        type=Path,
        default=DEFAULT_THREERETAINED_INPUT,
        help=(
            "Three-retained-predictor input TSV file "
            f"(default: {DEFAULT_THREERETAINED_INPUT})"
        ),
    )
    return parser.parse_args()


def compute_fourway_summary(
    summary: pd.DataFrame,
    oneretained_summary: pd.DataFrame,
    tworetained_summary: pd.DataFrame,
    threeretained_summary: pd.DataFrame,
) -> pd.DataFrame:
    main_lookup = summary.set_index(["group", "experiment", "year"])["mean_change"]
    single_lookup = oneretained_summary.set_index(
        ["group", "retained_predictor", "experiment", "year"]
    )["mean_change"]
    pair_lookup = tworetained_summary.set_index(
        ["group", "retained_predictor_pair", "experiment", "year"]
    )["mean_change"]
    triple_lookup = threeretained_summary.set_index(
        ["group", "retained_predictor_triple", "experiment", "year"]
    )["mean_change"]
    pair_keys = [
        ordered_combo_key(parts)
        for parts in itertools.combinations(RETAINED_PREDICTOR_ORDER, 2)
    ]
    triple_keys = [
        ordered_combo_key(parts)
        for parts in itertools.combinations(RETAINED_PREDICTOR_ORDER, 3)
    ]

    rows: list[dict[str, float | int | str]] = []
    for group in GROUP_ORDER:
        for experiment in FUTURE_EXPERIMENTS:
            years = summary[
                (summary["group"] == group) & (summary["experiment"] == experiment)
            ]["year"].tolist()
            for year in sorted(years):
                rows.append(
                    {
                        "group": group,
                        "experiment": experiment,
                        "year": int(year),
                        "mean_change": float(
                            main_lookup.loc[(group, experiment, year)]
                            - sum(
                                triple_lookup.loc[(group, key, experiment, year)]
                                for key in triple_keys
                            )
                            + sum(
                                pair_lookup.loc[(group, key, experiment, year)]
                                for key in pair_keys
                            )
                            - sum(
                                single_lookup.loc[(group, key, experiment, year)]
                                for key in RETAINED_PREDICTOR_ORDER
                            )
                        ),
                    }
                )

    return pd.DataFrame(rows)


def compute_fourway_method1_stats(
    summary: pd.DataFrame,
    oneretained_summary: pd.DataFrame,
    tworetained_summary: pd.DataFrame,
    threeretained_summary: pd.DataFrame,
) -> pd.DataFrame:
    fourway_summary = compute_fourway_summary(
        summary,
        oneretained_summary,
        tworetained_summary,
        threeretained_summary,
    )
    return compute_method1_stats(fourway_summary)


def compute_method1_stats_with_pvalues(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []

    for group in GROUP_ORDER:
        subset = summary[
            (summary["group"] == group)
            & (summary["experiment"].isin(FUTURE_EXPERIMENTS))
            & (summary["year"].between(2015, 2099))
        ].copy()
        if subset.empty:
            continue

        subset["year_centered"] = subset["year"] - 2015
        subset["is_ssp585"] = (subset["experiment"] == "ssp585").astype(int)
        subset["interaction"] = subset["year_centered"] * subset["is_ssp585"]

        design = sm.add_constant(subset[["year_centered", "is_ssp585", "interaction"]])
        fit = sm.OLS(subset["mean_change"], design).fit()
        rows.append(
            {
                "group": group,
                "slope_diff": float(fit.params["interaction"]),
                "p_value": float(fit.pvalues["interaction"]),
            }
        )

    return pd.DataFrame(rows)


def build_display_table(
    method1_stats: pd.DataFrame,
    oneretained_method1_stats: pd.DataFrame,
    twoway_interaction_stats: pd.DataFrame,
    threeway_interaction_stats: pd.DataFrame,
    fourway_method1_stats: pd.DataFrame,
) -> pd.DataFrame:
    two_way_lookup = twoway_interaction_stats.groupby("group")["effect_value"].sum()
    three_way_lookup = threeway_interaction_stats.groupby("group")["effect_value"].sum()
    fourway_lookup = fourway_method1_stats.set_index("group")
    total_lookup = method1_stats.set_index("group")

    rows: list[dict[str, float | str]] = []
    for group in GROUP_ORDER:
        rows.append(
            {
                "group": group,
                "effect_key": MAIN_EFFECT_KEY,
                "effect_value": float(total_lookup.loc[group, "slope_diff"]),
                "fill_text": "",
                "facecolor": MAIN_BAR_COLOR,
                "hatch": "",
                "p_value": float(total_lookup.loc[group, "p_value"]),
            }
        )
        predictor_rows = oneretained_method1_stats[
            oneretained_method1_stats["group"] == group
        ].set_index("retained_predictor")
        for predictor in RETAINED_PREDICTOR_ORDER:
            rows.append(
                {
                    "group": group,
                    "effect_key": predictor,
                    "effect_value": float(predictor_rows.loc[predictor, "slope_diff"]),
                    "fill_text": fill_text_for_combo((predictor,)),
                    "facecolor": PREDICTOR_BAR_FACE_COLOR,
                    "hatch": "",
                    "p_value": float("nan"),
                }
            )
        rows.append(
            {
                "group": group,
                "effect_key": INTERACTION_EFFECT_KEY,
                "effect_value": float(
                    two_way_lookup.loc[group]
                    + three_way_lookup.loc[group]
                    + fourway_lookup.loc[group, "slope_diff"]
                ),
                "fill_text": "",
                "facecolor": "white",
                "hatch": INTERACTION_EFFECT_HATCH,
                "p_value": float("nan"),
            }
        )

    return pd.DataFrame(rows)


def format_p_value_text(p_value: float) -> str:
    if pd.isna(p_value):
        return "p = NA"
    if p_value < 0.001:
        return "p < 0.001"
    return f"p = {round(p_value, 3):.3f}"


def plot_method_comparison(
    display_df: pd.DataFrame,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(10.2, 4.9), constrained_layout=True)
    fig.set_constrained_layout_pads(h_pad=0.08, w_pad=0.04)
    fig.supylabel(r"Tropical cyclone frequency trend difference (% year$^{-1}$)")

    effect_order = [
        MAIN_EFFECT_KEY,
        *RETAINED_PREDICTOR_ORDER,
        INTERACTION_EFFECT_KEY,
    ]
    panel_df = (
        display_df.pivot(index="group", columns="effect_key", values="effect_value")
        .reindex(GROUP_ORDER)
        .reindex(columns=effect_order)
    )
    effect_style = display_df.drop_duplicates("effect_key").set_index("effect_key")
    all_values = display_df["effect_value"].dropna()
    ymin = min(float(all_values.min()), 0.0) if not all_values.empty else -0.1
    ymax = max(float(all_values.max()), 0.0) if not all_values.empty else 0.1
    pad = max((ymax - ymin) * 0.18, max(abs(ymin), abs(ymax)) * 0.12, 0.02)
    y_lower = ymin - pad
    y_upper = 0.0 if ymax <= 0 else ymax + pad
    text_offset = max((ymax - ymin) * 0.06, 0.01)

    x = np.arange(len(GROUP_ORDER))
    offsets, bar_width = build_effect_offsets(effect_order)
    for offset, effect_key in zip(offsets, effect_order):
        bars = ax.bar(
            x + offset,
            panel_df[effect_key],
            width=bar_width,
            color=str(effect_style.loc[effect_key, "facecolor"]),
            edgecolor="black",
            linewidth=0.8,
            hatch=str(effect_style.loc[effect_key, "hatch"]),
        )
        fill_text = str(effect_style.loc[effect_key, "fill_text"])
        if fill_text:
            for bar in bars:
                add_centered_label_to_bar(ax, bar, fill_text)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(GROUP_ORDER)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    add_group_separators(ax)

    total_offset = float(offsets[0])
    total_series = (
        display_df[display_df["effect_key"] == MAIN_EFFECT_KEY]
        .set_index("group")
        .reindex(GROUP_ORDER)
    )
    for xpos, group in zip(x, GROUP_ORDER):
        row = total_series.loc[group]
        height = float(row["effect_value"])
        p_text = format_p_value_text(float(row["p_value"]))
        text_y = height + text_offset if height >= 0 else height - text_offset
        va = "bottom" if height >= 0 else "top"
        ax.text(
            xpos + total_offset,
            text_y,
            p_text,
            ha="center",
            va=va,
            fontsize=9,
        )

    ax.legend(
        handles=[
            Rectangle(
                (0, 0),
                1,
                1,
                facecolor=MAIN_BAR_COLOR,
                edgecolor="black",
                linewidth=0.8,
            ),
            *[
                LetterFillLegendItem(RETAINED_PREDICTOR_FILL_LETTERS[predictor])
                for predictor in RETAINED_PREDICTOR_ORDER
            ],
            CenteredHatchLegendItem(INTERACTION_EFFECT_HATCH),
        ],
        labels=[
            MAIN_EFFECT_LABEL,
            *[
                RETAINED_PREDICTOR_LABELS[predictor]
                for predictor in RETAINED_PREDICTOR_ORDER
            ],
            INTERACTION_EFFECT_LABEL,
        ],
        handler_map={
            LetterFillLegendItem: LetterFillLegendHandler(),
            CenteredHatchLegendItem: CenteredHatchLegendHandler(),
        },
        frameon=False,
        ncol=6,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        fontsize=9.5,
        handleheight=1.6,
        handlelength=2.0,
        handletextpad=0.7,
        borderpad=0.4,
        labelspacing=0.8,
        columnspacing=1.0,
    )
    ax.set_ylim(y_lower, y_upper)
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
    oneretained_df = load_oneretained_rate_table(args.oneretained_input)
    tworetained_df = load_tworetained_rate_table(args.tworetained_input)
    threeretained_df = load_threeretained_rate_table(args.threeretained_input)

    group_df = build_group_rate_table(df)
    baseline = compute_historical_baseline(group_df)
    change_df = apply_historical_baseline(group_df, baseline)
    summary = summarize_mean_change(change_df)
    method1_stats = compute_method1_stats_with_pvalues(summary)

    oneretained_change_df = compute_oneretained_change_table(oneretained_df, baseline)
    oneretained_summary = summarize_oneretained_mean_change(oneretained_change_df)
    oneretained_method1_stats = compute_oneretained_method1_stats(oneretained_summary)

    tworetained_change_df = compute_tworetained_change_table(tworetained_df, baseline)
    tworetained_summary = summarize_tworetained_mean_change(tworetained_change_df)
    tworetained_method1_stats = compute_tworetained_method1_stats(tworetained_summary)

    threeretained_change_df = compute_threeretained_change_table(
        threeretained_df, baseline
    )
    threeretained_summary = summarize_threeretained_mean_change(
        threeretained_change_df
    )
    threeretained_method1_stats = compute_threeretained_method1_stats(
        threeretained_summary
    )

    twoway_interaction_stats = compute_twoway_interaction_stats(
        oneretained_method1_stats, tworetained_method1_stats
    )
    threeway_interaction_stats = compute_threeway_interaction_stats(
        oneretained_method1_stats,
        tworetained_method1_stats,
        threeretained_method1_stats,
    )
    fourway_method1_stats = compute_fourway_method1_stats(
        summary,
        oneretained_summary,
        tworetained_summary,
        threeretained_summary,
    )
    display_df = build_display_table(
        method1_stats,
        oneretained_method1_stats,
        twoway_interaction_stats,
        threeway_interaction_stats,
        fourway_method1_stats,
    )

    plot_method_comparison(display_df, args.output)
    open_output(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
