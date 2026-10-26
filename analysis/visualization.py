"""Matplotlib figures with explicit labels and no external plotting dependency."""

from __future__ import annotations

from io import BytesIO

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

BLUE = "#2563A6"
TEAL = "#087F8C"
COLORS = [BLUE, TEAL, "#CC8546", "#8A61A6", "#6B7280"]


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.6)
    ax.set_axisbelow(True)


def _levels(series, order=None):
    observed = series.dropna().astype(str).unique().tolist()
    if order:
        return [str(x) for x in order if str(x) in observed]
    return sorted(observed)


def distribution_plot(data, column, types, kind="histogram", group=None, category_orders=None):
    """Plot a continuous distribution or categorical counts, optionally by group.

    Missing measurements are not plotted; the number plotted is in the title.
    A grouping variable's missing values are excluded from grouped plots only.
    """
    if column not in data:
        raise ValueError("Choose an existing variable.")
    orders = category_orders or {}
    frame = data[[column] + ([group] if group and group != column else [])].copy()
    if group:
        frame = frame.loc[frame[group].notna()]
    continuous = types.get(column) == "continuous"
    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    groups = _levels(frame[group], orders.get(group)) if group else [None]
    if len(groups) > 12:
        plt.close(fig)
        raise ValueError("Choose a grouping variable with at most 12 observed groups.")
    n = int(frame[column].notna().sum())
    if continuous:
        values = []
        for level in groups:
            subset = frame if level is None else frame.loc[frame[group].astype(str) == level]
            v = pd.to_numeric(subset[column], errors="coerce").dropna().to_numpy(float)
            values.append(v[np.isfinite(v)])
        if not any(len(v) for v in values):
            plt.close(fig)
            raise ValueError("No finite measurements to plot.")
        n = sum(len(v) for v in values)
        if kind == "boxplot":
            # Labels set separately for compatibility with Matplotlib 3.8+.
            boxes = ax.boxplot(values, patch_artist=True)
            for i, patch in enumerate(boxes["boxes"]):
                patch.set_facecolor(COLORS[i % len(COLORS)])
                patch.set_alpha(0.7)
            ax.set_xticks(range(1, len(groups) + 1), [str(g) if g is not None else column for g in groups])
            ax.set_ylabel(column)
            if group:
                ax.set_xlabel(group)
        else:
            all_values = np.concatenate([v for v in values if len(v)])
            bins = np.histogram_bin_edges(all_values, bins="auto")
            if len(bins) > 60:
                bins = np.linspace(all_values.min(), all_values.max(), 41)
            for i, (label, v) in enumerate(zip(groups, values)):
                ax.hist(v, bins=bins, alpha=0.65 if group else 0.9,
                        color=COLORS[i % len(COLORS)], edgecolor="white",
                        label=str(label) if group else None)
            ax.set_xlabel(column)
            ax.set_ylabel("Number of observations")
            if group:
                ax.legend(title=group, frameon=False)
    else:
        levels = _levels(frame[column], orders.get(column))
        if not levels:
            plt.close(fig)
            raise ValueError("No observed categories to plot.")
        if len(levels) > 40:
            plt.close(fig)
            raise ValueError("This variable has more than 40 levels; use its frequency table.")
        x = np.arange(len(levels))
        width = 0.8 / max(1, len(groups))
        for i, g in enumerate(groups):
            subset = frame if g is None else frame.loc[frame[group].astype(str) == g]
            counts = subset[column].dropna().astype(str).value_counts().reindex(levels, fill_value=0)
            ax.bar(x + (i - (len(groups) - 1) / 2) * width, counts,
                   width=width, color=COLORS[i % len(COLORS)], label=str(g) if group else None)
        ax.set_xticks(x, levels, rotation=30 if len(levels) > 4 else 0, ha="right" if len(levels) > 4 else "center")
        ax.set_xlabel(column)
        ax.set_ylabel("Number of observations")
        if group:
            ax.legend(title=group, frameon=False)
    ax.set_title(f"{column} · {n:,} observed measurements", loc="left", fontsize=12, pad=14)
    _style(ax)
    return fig


def forest_plot(model):
    """Plot all estimated non-intercept terms and their 95% confidence intervals."""
    if not model.get("success") or model["terms"].empty:
        raise ValueError("A successfully fitted model is needed for a forest plot.")
    terms = model["terms"].copy().iloc[::-1]
    logistic = model.get("info", {}).get("family") == "logistic"
    adjusted = len(model["terms"]["variable"].unique()) > 1
    finite = np.isfinite(terms[["estimate", "ci_lower", "ci_upper"]].astype(float)).all(axis=1)
    if not finite.all() or (logistic and (terms["ci_lower"] <= 0).any()):
        raise ValueError("The model has an interval that cannot be displayed reliably.")
    fig, ax = plt.subplots(figsize=(9, max(3.5, 0.48 * len(terms) + 1.6)), layout="constrained")
    estimates = terms["estimate"].to_numpy(float)
    lower = terms["ci_lower"].to_numpy(float)
    upper = terms["ci_upper"].to_numpy(float)
    y = np.arange(len(terms))
    ax.hlines(y, lower, upper, color=TEAL, linewidth=1.4)
    ax.vlines(lower, y-0.07, y+0.07, color=TEAL, linewidth=1.2)
    ax.vlines(upper, y-0.07, y+0.07, color=TEAL, linewidth=1.2)
    ax.scatter(estimates, y, color=BLUE, s=28, zorder=3)
    ax.axvline(1 if logistic else 0, color="#9CA3AF", linestyle="--", linewidth=1)
    if logistic:
        ax.set_xscale("log")
    ax.set_yticks(y, terms["term"].astype(str).tolist())
    prefix = "Adjusted " if adjusted else "Unadjusted "
    ax.set_xlabel(prefix + ("odds ratio (95% CI; logarithmic scale)" if logistic else "coefficient (95% CI)"))
    ax.set_title(f"{prefix}associations · n = {model['n_complete']:,}", loc="left", pad=14, fontsize=12)
    _style(ax)
    ax.grid(axis="x", color="#E5E7EB", linewidth=0.6)
    return fig


def diagnostic_plot(model):
    """Basic visual diagnostics; all plots describe the model's fitting sample."""
    if not model.get("success"):
        raise ValueError("Fit a model before requesting diagnostics.")
    d = model["diagnostics"]
    if d.empty:
        raise ValueError("No diagnostics are available for this fit.")
    logistic = model.get("info", {}).get("family") == "logistic"
    if logistic:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), layout="constrained")
        observed = d["observed"].to_numpy(float)
        fitted = d["fitted"].to_numpy(float)
        bins = min(10, max(2, len(d) // 25), int(pd.Series(fitted).nunique()))
        cal = pd.DataFrame({"observed": observed, "fitted": fitted})
        if bins >= 2:
            cal["bin"] = pd.qcut(cal["fitted"], q=bins, duplicates="drop")
            grouped = cal.groupby("bin", observed=True).agg(predicted=("fitted", "mean"), actual=("observed", "mean"), n=("observed", "size"))
        else:
            grouped = pd.DataFrame({"predicted": [fitted.mean()], "actual": [observed.mean()], "n": [len(d)]})
        axes[0].plot([0, 1], [0, 1], "--", color="#9CA3AF", linewidth=1)
        axes[0].plot(grouped["predicted"], grouped["actual"], "o-", color=TEAL)
        axes[0].set(xlabel="Mean fitted probability", ylabel="Observed event proportion", xlim=(0,1), ylim=(0,1), title="Apparent calibration (grouped)")
        axes[1].hist(fitted[observed == 0], bins=np.linspace(0,1,21), alpha=0.65, color=BLUE, label="Non-event")
        axes[1].hist(fitted[observed == 1], bins=np.linspace(0,1,21), alpha=0.65, color=TEAL, label="Event")
        axes[1].set(xlabel="Fitted probability", ylabel="Number of observations", title="Fitted probabilities by outcome")
        axes[1].legend(frameon=False)
        fig.suptitle("Fitting-sample diagnostics · not external or internal validation", fontsize=11)
    else:
        fig, axes = plt.subplots(2, 2, figsize=(10, 7.5), layout="constrained")
        axes = axes.ravel()
        axes[0].scatter(d["fitted"], d["residual"], s=14, alpha=0.5, color=BLUE)
        axes[0].axhline(0, color="#9CA3AF", linestyle="--")
        axes[0].set(xlabel="Fitted value", ylabel="Residual", title="Residuals versus fitted")
        residuals = d.get("std_residual", d["residual"]).to_numpy(float)
        residuals = residuals[np.isfinite(residuals)]
        (theoretical, ordered), (slope, intercept, _) = stats.probplot(residuals, dist="norm")
        axes[1].scatter(theoretical, ordered, s=14, alpha=0.5, color=BLUE)
        axes[1].plot(theoretical, intercept+slope*theoretical, color=TEAL)
        axes[1].set(xlabel="Theoretical normal quantile", ylabel="Ordered standardized residual", title="Normal Q–Q plot")
        std = d.get("std_residual", d["residual"]).to_numpy(float)
        axes[2].scatter(d["fitted"], np.sqrt(np.abs(std)), s=14, alpha=0.5, color=TEAL)
        axes[2].set(xlabel="Fitted value", ylabel="√ |standardized residual|", title="Scale–location")
        if "cooks_distance" in d:
            axes[3].scatter(d["row_number"], d["cooks_distance"], s=14, alpha=0.6, color=TEAL)
            axes[3].set(xlabel="Row position in cleaned data", ylabel="Cook’s distance", title="Influence by observation")
        else:
            axes[3].axis("off")
        fig.suptitle("Linear regression diagnostics · review patterns and study design together", fontsize=11)
    for ax in np.asarray(axes).ravel():
        _style(ax)
    return fig


def figure_bytes(fig, format="png"):
    """Serialize a figure; caller owns and should close the Figure afterwards."""
    if format not in {"png", "svg"}:
        raise ValueError("Supported figure formats are PNG and SVG.")
    buf = BytesIO()
    fig.savefig(buf, format=format, dpi=300, bbox_inches="tight", facecolor="white")
    return buf.getvalue()
