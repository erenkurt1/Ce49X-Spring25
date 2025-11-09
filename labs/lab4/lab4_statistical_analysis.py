"""
Lab 4: Statistical Analysis
Descriptive Statistics and Probability Distributions
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from scipy import stats
from scipy.stats import binom, expon, norm, poisson, uniform
from io import StringIO

sns.set_theme(style="whitegrid", context="talk")


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "datasets"
OUTPUT_DIR = Path(__file__).resolve().parent


BERNOULLI_PASS_PROB = 0.95
BINOMIAL_DEFECT_RATE = 0.05
BINOMIAL_TRIALS = 100
POISSON_EVENTS_PER_INTERVAL = 10
NORMAL_MEAN = 250
NORMAL_STD = 15
EXPONENTIAL_MEAN = 1000
UNIFORM_LOW = 20
UNIFORM_HIGH = 40


def load_data(file_path: str) -> pd.DataFrame:
    """Load dataset from the repository-level datasets folder."""
    dataset_path = DATA_DIR / file_path
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset '{file_path}' not found in '{DATA_DIR}'. Ensure the datasets directory exists."
        )
    return pd.read_csv(dataset_path)


def calculate_descriptive_stats(data: pd.DataFrame, column: str = "strength_mpa") -> pd.DataFrame:
    """Calculate descriptive statistics for a numeric column."""
    series = data[column].dropna().astype(float)
    stats_dict: Dict[str, Union[float, int]] = {
        "count": series.count(),
        "mean": series.mean(),
        "median": series.median(),
        "mode": series.mode().iloc[0] if not series.mode().empty else math.nan,
        "min": series.min(),
        "q1": series.quantile(0.25),
        "q3": series.quantile(0.75),
        "max": series.max(),
        "variance": series.var(),
        "std_dev": series.std(),
        "range": series.max() - series.min(),
        "iqr": stats.iqr(series),
        "skewness": stats.skew(series),
        "kurtosis": stats.kurtosis(series),
    }
    return pd.DataFrame(stats_dict, index=[column]).T


def plot_distribution(
    data: pd.DataFrame,
    column: str,
    title: str,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot histogram with KDE and mark mean/median/mode."""
    series = data[column].dropna().astype(float)
    mean_val = series.mean()
    median_val = series.median()
    mode_val = series.mode().iloc[0] if not series.mode().empty else math.nan

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(series, kde=True, ax=ax, color="#4C72B0", edgecolor="white")

    for value, label, color in [
        (mean_val, "Mean", "#55A868"),
        (median_val, "Median", "#C44E52"),
        (mode_val, "Mode", "#8172B2"),
    ]:
        if not math.isnan(value):
            ax.axvline(value, color=color, linestyle="--", linewidth=2, label=f"{label}: {value:.2f}")

    ax.set_title(title)
    ax.set_xlabel(column.replace("_", " ").title())
    ax.set_ylabel("Frequency")
    ax.legend()

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_spread_bands(
    data: pd.DataFrame,
    column: str,
    title: str,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot distribution highlighting +/- 1, 2, 3 sigma regions."""
    series = data[column].dropna().astype(float)
    mean_val = series.mean()
    std_val = series.std()

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(series, bins=20, stat="density", color="#4C72B0", edgecolor="white", ax=ax)
    sns.kdeplot(series, ax=ax, color="#1B9E77", linewidth=2)

    sigma_bands = [
        (1, "#E6550D", 0.15),
        (2, "#3182BD", 0.10),
        (3, "#31A354", 0.05),
    ]
    for multiplier, color, alpha in sigma_bands:
        lower = mean_val - multiplier * std_val
        upper = mean_val + multiplier * std_val
        ax.axvspan(lower, upper, color=color, alpha=alpha, label=f"+/-{multiplier} sigma")

    ax.axvline(mean_val, color="#C51B7D", linestyle="--", linewidth=2, label=f"Mean: {mean_val:.2f}")
    ax.set_title(title)
    ax.set_xlabel(column.replace("_", " ").title())
    ax.set_ylabel("Density")
    ax.legend()

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def fit_distribution(
    data: pd.DataFrame,
    column: str,
    distribution_type: str = "normal",
) -> Tuple[float, float]:
    """Fit a distribution to data (currently supports normal only)."""
    series = data[column].dropna().astype(float)
    if distribution_type != "normal":
        raise ValueError("Only normal distribution fitting is implemented.")
    mu, sigma = norm.fit(series)
    return mu, sigma


def calculate_probability_binomial(n: int, p: float, k: Union[int, List[int]]) -> float:
    """Calculate binomial probability for exact outcomes or list of outcomes."""
    if isinstance(k, list):
        return float(sum(binom.pmf(kk, n, p) for kk in k))
    return float(binom.pmf(k, n, p))


def calculate_probability_normal(
    mean: float,
    std: float,
    x_lower: Optional[float] = None,
    x_upper: Optional[float] = None,
) -> float:
    """Calculate cumulative normal probability between bounds."""
    if x_lower is None and x_upper is None:
        raise ValueError("Provide at least one bound.")
    dist = norm(loc=mean, scale=std)
    if x_lower is None:
        return float(dist.cdf(x_upper))
    if x_upper is None:
        return float(1 - dist.cdf(x_lower))
    return float(dist.cdf(x_upper) - dist.cdf(x_lower))


def calculate_probability_poisson(lambda_param: float, k: Union[int, List[int]]) -> float:
    """Calculate Poisson probability for exact outcomes or list of outcomes."""
    if isinstance(k, list):
        return float(sum(poisson.pmf(kk, lambda_param) for kk in k))
    return float(poisson.pmf(k, lambda_param))


def calculate_probability_exponential(mean: float, x: float, survival: bool = False) -> float:
    """Calculate exponential distribution probabilities."""
    scale = mean
    if survival:
        return float(expon.sf(x, scale=scale))
    return float(expon.cdf(x, scale=scale))


def apply_bayes_theorem(prior: float, sensitivity: float, specificity: float) -> float:
    """Compute posterior probability using Bayes' theorem."""
    true_positive = sensitivity * prior
    false_positive = (1 - specificity) * (1 - prior)
    return true_positive / (true_positive + false_positive)


def plot_material_comparison(
    data: pd.DataFrame,
    column: str,
    group_column: str,
    title: str,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Create comparative box plot for material strengths."""
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=data, x=group_column, y=column, palette="Set2", ax=ax)
    sns.stripplot(data=data, x=group_column, y=column, color="black", alpha=0.4, ax=ax)
    ax.set_title(title)
    ax.set_xlabel(group_column.replace("_", " ").title())
    ax.set_ylabel(column.replace("_", " ").title())

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_distribution_fitting(
    data: pd.DataFrame,
    column: str,
    fitted_params: Tuple[float, float],
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Compare empirical data with fitted normal distribution."""
    series = data[column].dropna().astype(float)
    mu, sigma = fitted_params
    synthetic = np.random.normal(loc=mu, scale=sigma, size=10_000)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(series, bins=20, stat="density", color="#4C72B0", alpha=0.6, ax=ax, label="Observed")
    sns.kdeplot(synthetic, ax=ax, color="#DD8452", linewidth=2, label="Synthetic (Fitted)")

    x_vals = np.linspace(series.min(), series.max(), 200)
    ax.plot(x_vals, norm.pdf(x_vals, mu, sigma), color="#55A868", linestyle="--", linewidth=2, label="Fitted PDF")

    ax.set_title(f"Distribution Fitting for {column.replace('_', ' ').title()}")
    ax.set_xlabel(column.replace("_", " ").title())
    ax.set_ylabel("Density")
    ax.legend()

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def create_probability_distributions_plot(
    save_path: Optional[Path] = None,
) -> Tuple[plt.Figure, Dict[str, Dict[str, float]]]:
    """Visualize distributions and collect sample/theoretical statistics."""
    rng = np.random.default_rng(42)

    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    fig.subplots_adjust(hspace=0.35)

    distribution_stats: Dict[str, Dict[str, float]] = {}

    # --- Row 1: Bernoulli (discrete) and Uniform (continuous) ---
    # Bernoulli parameters
    bernoulli_samples = rng.binomial(1, BERNOULLI_PASS_PROB, size=2000)
    bernoulli_counts = np.bincount(bernoulli_samples, minlength=2) / bernoulli_samples.size
    bernoulli_x = np.arange(2)
    bernoulli_pmf_theory = binom.pmf(bernoulli_x, 1, BERNOULLI_PASS_PROB)
    bernoulli_cdf_sample = np.cumsum(bernoulli_counts)
    bernoulli_cdf_theory = binom.cdf(bernoulli_x, 1, BERNOULLI_PASS_PROB)

    ax_pmf = axes[0, 0]
    width = 0.35
    ax_pmf.bar(bernoulli_x - width / 2, bernoulli_counts, width=width, label="Sample PMF", color="#4C72B0")
    ax_pmf.bar(bernoulli_x + width / 2, bernoulli_pmf_theory, width=width, label="Theoretical PMF", color="#55A868")
    ax_pmf.set_title("Bernoulli Distribution (p=0.95)")
    ax_pmf.set_xlabel("Outcome")
    ax_pmf.set_ylabel("Probability")
    ax_pmf.set_xticks(bernoulli_x)
    ax_pmf.set_ylim(0, 1.05)

    ax_cdf = ax_pmf.twinx()
    ax_cdf.step(bernoulli_x, bernoulli_cdf_sample, where="post", label="Sample CDF", color="#C44E52")
    ax_cdf.step(bernoulli_x, bernoulli_cdf_theory, where="post", label="Theoretical CDF", color="#8172B2", linestyle="--")
    ax_cdf.set_ylabel("Cumulative Probability")
    ax_cdf.set_ylim(0, 1.05)

    handles, labels = ax_pmf.get_legend_handles_labels()
    handles2, labels2 = ax_cdf.get_legend_handles_labels()
    ax_pmf.legend(handles + handles2, labels + labels2, loc="upper left")

    distribution_stats["Bernoulli (p=0.95)"] = {
        "sample_mean": float(bernoulli_samples.mean()),
        "sample_variance": float(bernoulli_samples.var(ddof=1)),
        "theoretical_mean": BERNOULLI_PASS_PROB,
        "theoretical_variance": BERNOULLI_PASS_PROB * (1 - BERNOULLI_PASS_PROB),
    }

    # Uniform distribution
    uniform_samples = rng.uniform(UNIFORM_LOW, UNIFORM_HIGH, size=4000)
    ax_pdf = axes[0, 1]
    ax_pdf.hist(uniform_samples, bins=30, density=True, alpha=0.5, color="#4C72B0", label="Sample PDF")
    x_uniform = np.linspace(UNIFORM_LOW, UNIFORM_HIGH, 400)
    ax_pdf.plot(x_uniform, uniform.pdf(x_uniform, loc=UNIFORM_LOW, scale=UNIFORM_HIGH - UNIFORM_LOW), color="#55A868", linewidth=2, label="Theoretical PDF")
    ax_pdf.set_title(f"Uniform Distribution ({UNIFORM_LOW}, {UNIFORM_HIGH})")
    ax_pdf.set_xlabel("Value")
    ax_pdf.set_ylabel("Density")

    ax_cdf_uniform = ax_pdf.twinx()
    sorted_uniform = np.sort(uniform_samples)
    ecdf_uniform = np.arange(1, sorted_uniform.size + 1) / sorted_uniform.size
    ax_cdf_uniform.plot(sorted_uniform, ecdf_uniform, color="#C44E52", label="Sample CDF")
    ax_cdf_uniform.plot(x_uniform, uniform.cdf(x_uniform, loc=UNIFORM_LOW, scale=UNIFORM_HIGH - UNIFORM_LOW), color="#8172B2", linestyle="--", label="Theoretical CDF")
    ax_cdf_uniform.set_ylabel("Cumulative Probability")

    handles, labels = ax_pdf.get_legend_handles_labels()
    handles2, labels2 = ax_cdf_uniform.get_legend_handles_labels()
    ax_pdf.legend(handles + handles2, labels + labels2, loc="upper left")

    distribution_stats[f"Uniform ({UNIFORM_LOW}, {UNIFORM_HIGH})"] = {
        "sample_mean": float(uniform_samples.mean()),
        "sample_variance": float(uniform_samples.var(ddof=1)),
        "theoretical_mean": (UNIFORM_LOW + UNIFORM_HIGH) / 2,
        "theoretical_variance": (UNIFORM_HIGH - UNIFORM_LOW) ** 2 / 12,
    }

    # --- Row 2: Binomial and Normal ---
    n_trials, defect_rate = BINOMIAL_TRIALS, BINOMIAL_DEFECT_RATE
    binomial_samples = rng.binomial(n_trials, defect_rate, size=4000)
    max_k = binomial_samples.max()
    bin_edges = np.arange(max_k + 2) - 0.5
    ax_bin_pmf = axes[1, 0]
    ax_bin_pmf.hist(binomial_samples, bins=bin_edges, density=True, alpha=0.5, color="#4C72B0", label="Sample PMF")
    support = np.arange(0, max_k + 1)
    ax_bin_pmf.plot(support, binom.pmf(support, n_trials, defect_rate), marker="o", linestyle="--", color="#55A868", label="Theoretical PMF")
    ax_bin_pmf.set_title(f"Binomial Distribution (n={n_trials}, p={defect_rate})")
    ax_bin_pmf.set_xlabel("Number of Defects")
    ax_bin_pmf.set_ylabel("Probability")
    ax_bin_pmf.set_xlim(-0.5, min(max_k + 0.5, 20))

    ax_bin_cdf = ax_bin_pmf.twinx()
    counts = np.bincount(binomial_samples, minlength=max_k + 1) / binomial_samples.size
    cdf_sample = np.cumsum(counts)
    trimmed_support = support[: min(len(support), 21)]
    ax_bin_cdf.step(trimmed_support, cdf_sample[: len(trimmed_support)], where="post", color="#C44E52", label="Sample CDF")
    ax_bin_cdf.step(trimmed_support, binom.cdf(trimmed_support, n_trials, defect_rate), where="post", color="#8172B2", linestyle="--", label="Theoretical CDF")
    ax_bin_cdf.set_ylabel("Cumulative Probability")
    ax_bin_pmf.legend(loc="upper right")
    ax_bin_cdf.legend(loc="upper left")

    distribution_stats[f"Binomial (n={n_trials}, p={defect_rate})"] = {
        "sample_mean": float(binomial_samples.mean()),
        "sample_variance": float(binomial_samples.var(ddof=1)),
        "theoretical_mean": n_trials * defect_rate,
        "theoretical_variance": n_trials * defect_rate * (1 - defect_rate),
    }

    # Normal distribution
    normal_mean, normal_std = NORMAL_MEAN, NORMAL_STD
    normal_samples = rng.normal(normal_mean, normal_std, size=4000)
    ax_norm_pdf = axes[1, 1]
    ax_norm_pdf.hist(normal_samples, bins=40, density=True, alpha=0.5, color="#4C72B0", label="Sample PDF")
    x_norm = np.linspace(normal_mean - 4 * normal_std, normal_mean + 4 * normal_std, 400)
    ax_norm_pdf.plot(x_norm, norm.pdf(x_norm, normal_mean, normal_std), color="#55A868", linewidth=2, label="Theoretical PDF")
    ax_norm_pdf.set_title(f"Normal Distribution (μ={normal_mean}, σ={normal_std})")
    ax_norm_pdf.set_xlabel("Strength (MPa)")
    ax_norm_pdf.set_ylabel("Density")

    ax_norm_cdf = ax_norm_pdf.twinx()
    sorted_norm = np.sort(normal_samples)
    ecdf_norm = np.arange(1, sorted_norm.size + 1) / sorted_norm.size
    ax_norm_cdf.plot(sorted_norm, ecdf_norm, color="#C44E52", label="Sample CDF")
    ax_norm_cdf.plot(x_norm, norm.cdf(x_norm, normal_mean, normal_std), color="#8172B2", linestyle="--", label="Theoretical CDF")
    ax_norm_cdf.set_ylabel("Cumulative Probability")

    handles, labels = ax_norm_pdf.get_legend_handles_labels()
    handles2, labels2 = ax_norm_cdf.get_legend_handles_labels()
    ax_norm_pdf.legend(handles + handles2, labels + labels2, loc="upper left")

    distribution_stats[f"Normal (mean={normal_mean}, std={normal_std})"] = {
        "sample_mean": float(normal_samples.mean()),
        "sample_variance": float(normal_samples.var(ddof=1)),
        "theoretical_mean": normal_mean,
        "theoretical_variance": normal_std**2,
    }

    # --- Row 3: Poisson and Exponential ---
    poisson_lambda = POISSON_EVENTS_PER_INTERVAL
    poisson_samples = rng.poisson(poisson_lambda, size=4000)
    max_k_poisson = poisson_samples.max()
    bins_poisson = np.arange(max_k_poisson + 2) - 0.5
    ax_pois_pmf = axes[2, 0]
    ax_pois_pmf.hist(poisson_samples, bins=bins_poisson, density=True, alpha=0.5, color="#4C72B0", label="Sample PMF")
    support_poisson = np.arange(0, max_k_poisson + 1)
    ax_pois_pmf.plot(support_poisson, poisson.pmf(support_poisson, poisson_lambda), color="#55A868", linestyle="--", marker="o", label="Theoretical PMF")
    ax_pois_pmf.set_title(f"Poisson Distribution (λ={poisson_lambda})")
    ax_pois_pmf.set_xlabel("Events per Interval")
    ax_pois_pmf.set_ylabel("Probability")

    ax_pois_cdf = ax_pois_pmf.twinx()
    counts_poisson = np.bincount(poisson_samples, minlength=max_k_poisson + 1) / poisson_samples.size
    cdf_poisson = np.cumsum(counts_poisson)
    ax_pois_cdf.step(support_poisson, cdf_poisson, where="post", color="#C44E52", label="Sample CDF")
    ax_pois_cdf.step(support_poisson, poisson.cdf(support_poisson, poisson_lambda), where="post", color="#8172B2", linestyle="--", label="Theoretical CDF")
    ax_pois_cdf.set_ylabel("Cumulative Probability")
    ax_pois_pmf.legend(loc="upper right")
    ax_pois_cdf.legend(loc="upper left")

    distribution_stats[f"Poisson (lambda={poisson_lambda})"] = {
        "sample_mean": float(poisson_samples.mean()),
        "sample_variance": float(poisson_samples.var(ddof=1)),
        "theoretical_mean": poisson_lambda,
        "theoretical_variance": poisson_lambda,
    }

    exponential_mean = EXPONENTIAL_MEAN
    exponential_samples = rng.exponential(exponential_mean, size=4000)
    ax_exp_pdf = axes[2, 1]
    ax_exp_pdf.hist(exponential_samples, bins=40, density=True, alpha=0.5, color="#4C72B0", label="Sample PDF")
    x_exp = np.linspace(0, np.quantile(exponential_samples, 0.99), 400)
    ax_exp_pdf.plot(x_exp, expon.pdf(x_exp, scale=exponential_mean), color="#55A868", linewidth=2, label="Theoretical PDF")
    ax_exp_pdf.set_title(f"Exponential Distribution (mean={exponential_mean})")
    ax_exp_pdf.set_xlabel("Time")
    ax_exp_pdf.set_ylabel("Density")

    ax_exp_cdf = ax_exp_pdf.twinx()
    sorted_exp = np.sort(exponential_samples)
    ecdf_exp = np.arange(1, sorted_exp.size + 1) / sorted_exp.size
    ax_exp_cdf.plot(sorted_exp, ecdf_exp, color="#C44E52", label="Sample CDF")
    ax_exp_cdf.plot(x_exp, expon.cdf(x_exp, scale=exponential_mean), color="#8172B2", linestyle="--", label="Theoretical CDF")
    ax_exp_cdf.set_ylabel("Cumulative Probability")

    handles, labels = ax_exp_pdf.get_legend_handles_labels()
    handles2, labels2 = ax_exp_cdf.get_legend_handles_labels()
    ax_exp_pdf.legend(handles + handles2, labels + labels2, loc="upper right")

    distribution_stats[f"Exponential (mean={exponential_mean})"] = {
        "sample_mean": float(exponential_samples.mean()),
        "sample_variance": float(exponential_samples.var(ddof=1)),
        "theoretical_mean": exponential_mean,
        "theoretical_variance": exponential_mean**2,
    }

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig, distribution_stats


def plot_probability_tree(
    prior: float,
    sensitivity: float,
    specificity: float,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Visualize Bayes' theorem diagnostic tree with balanced layout."""
    posterior = apply_bayes_theorem(prior, sensitivity, specificity)
    false_alarm = 1 - specificity
    true_positive = sensitivity * prior
    false_negative = (1 - sensitivity) * prior
    true_negative = specificity * (1 - prior)
    false_positive = false_alarm * (1 - prior)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 12)
    ax.set_ylim(-1, 8.5)
    ax.axis("off")

    box_width = 2.1
    box_height = 1.0

    def draw_box(x: float, y: float, text: str, color: str) -> None:
        box = FancyBboxPatch(
            (x, y),
            box_width,
            box_height,
            boxstyle="round,pad=0.3",
            linewidth=1.5,
            edgecolor="#4F4F4F",
            facecolor=color,
        )
        ax.add_patch(box)
        ax.text(x + box_width / 2, y + box_height / 2, text, ha="center", va="center", fontsize=11, fontweight="bold")

    def draw_arrow(start: Tuple[float, float], end: Tuple[float, float], label: str) -> None:
        arrow = FancyArrowPatch(
            start,
            end,
            arrowstyle="->",
            mutation_scale=15,
            linewidth=2,
            color="#4F4F4F",
        )
        ax.add_patch(arrow)
        mid_x = (start[0] + end[0]) / 2
        mid_y = (start[1] + end[1]) / 2
        if label:
            ax.text(mid_x, mid_y + 0.3, label, ha="center", va="center", fontsize=11, color="#1F4E79")

    root_x = 1
    branch_x = 4
    outcome_x = 7
    posterior_x = 10

    root_y = 3.5
    damage_y = 6.0
    no_damage_y = 2.0

    draw_box(root_x, root_y, "Structure", "#E8F1FA")
    draw_box(branch_x, damage_y, f"Damage\n{prior*100:.1f}%", "#FDE2E2")
    draw_box(branch_x, no_damage_y, f"No Damage\n{(1 - prior)*100:.1f}%", "#E2F3E4")

    root_center = (root_x + box_width, root_y + box_height / 2)
    damage_center = (branch_x, damage_y + box_height / 2)
    no_damage_center = (branch_x, no_damage_y + box_height / 2)

    draw_arrow(root_center, (branch_x, damage_center[1]), "Prior 5%")
    draw_arrow(root_center, (branch_x, no_damage_center[1]), "Prior 95%")

    outcomes = [
        ("Positive Test\nTP {0:.2f}%".format(true_positive * 100), "#FCE4EC"),
        ("Negative Test\nFN {0:.2f}%".format(false_negative * 100), "#FFF4E5"),
        ("Positive Test\nFP {0:.2f}%".format(false_positive * 100), "#FFF8DC"),
        ("Negative Test\nTN {0:.2f}%".format(true_negative * 100), "#E8F8F5"),
    ]
    outcome_y_positions = [6.5, 4.5, 2.5, 0.5]

    for (text, color), y in zip(outcomes, outcome_y_positions):
        draw_box(outcome_x, y, text, color)

    tp_center = (outcome_x, outcome_y_positions[0] + box_height / 2)
    fn_center = (outcome_x, outcome_y_positions[1] + box_height / 2)
    fp_center = (outcome_x, outcome_y_positions[2] + box_height / 2)
    tn_center = (outcome_x, outcome_y_positions[3] + box_height / 2)

    draw_arrow((branch_x + box_width, damage_center[1]), tp_center, "Sensitivity 95%")
    draw_arrow((branch_x + box_width, damage_center[1]), fn_center, "Miss 5%")
    draw_arrow((branch_x + box_width, no_damage_center[1]), fp_center, "False alarm 10%")
    draw_arrow((branch_x + box_width, no_damage_center[1]), tn_center, "Specificity 90%")

    posterior_y = 6.4
    draw_box(posterior_x, posterior_y, f"Posterior\nP(Damage | +)\n{posterior*100:.2f}%", "#D9EDF7")
    draw_arrow((outcome_x + box_width, tp_center[1]), (posterior_x, posterior_y + box_height / 2), "")

    ax.text(
        posterior_x - 0.4,
        2.5,
        "Insight:\nEven with a sensitive test,\nfalse positives lower certainty.",
        fontsize=12,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#F4F6F6", edgecolor="#4F4F4F"),
    )

    ax.set_title("Bayes' Theorem Probability Tree: Structural Damage Test", fontsize=16, pad=20)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def create_statistical_report(
    concrete_stats: pd.DataFrame,
    structural_stats: pd.Series,
    probability_results: Dict[str, float],
    distribution_stats: Optional[Dict[str, Dict[str, float]]],
    material_summary: Optional[pd.DataFrame],
    posterior_probability: float,
    output_file: Union[str, Path] = "lab4_statistical_report.txt",
) -> Path:
    """Generate textual summary report."""
    output_path = OUTPUT_DIR / output_file
    with output_path.open("w", encoding="utf-8") as report:
        report.write("Lab 4 Statistical Analysis Report\n")
        report.write("=" * 40 + "\n\n")

        report.write("Concrete Strength Descriptive Statistics:\n")
        report.write(concrete_stats.to_string(float_format=lambda x: f"{x:0.4f}") + "\n\n")

        report.write("Structural Load Summary Statistics:\n")
        report.write(structural_stats.to_frame().T.to_string(float_format=lambda x: f"{x:0.4f}") + "\n\n")

        report.write("Probability Scenario Results:\n")
        for name, value in probability_results.items():
            report.write(f"- {name}: {value:.4f}\n")
        report.write("\n")

        if distribution_stats:
            report.write("Distribution Sample vs. Theoretical Statistics:\n")
            for name, stats_dict in distribution_stats.items():
                report.write(f"- {name}:\n")
                report.write(f"    Sample mean: {stats_dict['sample_mean']:.4f}\n")
                report.write(f"    Theoretical mean: {stats_dict['theoretical_mean']:.4f}\n")
                report.write(f"    Sample variance: {stats_dict['sample_variance']:.4f}\n")
                report.write(f"    Theoretical variance: {stats_dict['theoretical_variance']:.4f}\n")
            report.write("\n")

        if material_summary is not None and not material_summary.empty:
            report.write("Material Strength Summary (mean and standard deviation):\n")
            report.write(
                material_summary[["material", "count", "mean", "std"]]
                .to_string(index=False, float_format=lambda x: f"{x:0.4f}")
                + "\n\n"
            )

        report.write("Bayes' Theorem Posterior Probability:\n")
        report.write(f"P(Damage | Positive Test) = {posterior_probability:.4f}\n\n")

        report.write("Engineering Interpretation:\n")
        report.write(
            "- Concrete strengths show central tendency around the mean with variability captured by standard deviation.\n"
        )
        report.write(
            "- Probability analysis informs quality control (binomial), load forecasting (Poisson), "
            "material reliability (Normal), and component lifetime (Exponential).\n"
        )
        report.write(
            "- Despite high test sensitivity, the posterior probability underscores the impact of false positives "
            "on structural damage assessments.\n"
        )
    return output_path


def create_statistical_summary_dashboard(
    concrete_data: pd.DataFrame,
    material_data: pd.DataFrame,
    material_summary: Optional[pd.DataFrame] = None,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Create a 2x2 dashboard of summary plots."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    sns.histplot(concrete_data["strength_mpa"], kde=True, ax=axes[0, 0], color="#4C72B0")
    axes[0, 0].set_title("Concrete Strength Distribution")
    axes[0, 0].set_xlabel("Strength (MPa)")

    sns.boxplot(x="mix_type", y="strength_mpa", data=concrete_data, ax=axes[0, 1], palette="Set3")
    axes[0, 1].set_title("Concrete Strength by Mix Type")
    axes[0, 1].set_xlabel("Mix Type")
    axes[0, 1].set_ylabel("Strength (MPa)")

    if material_summary is None:
        material_summary = (
            material_data.groupby("material")["strength_mpa"].agg(["mean", "std"]).reset_index()
        )
    axes[1, 0].bar(material_summary["material"], material_summary["mean"], yerr=material_summary["std"], capsize=6)
    axes[1, 0].set_title("Material Strength Comparison")
    axes[1, 0].set_ylabel("Mean Strength (MPa)")

    sns.violinplot(data=material_data, x="material", y="strength_mpa", ax=axes[1, 1], palette="Pastel1")
    axes[1, 1].set_title("Material Strength Distribution (Violin)")
    axes[1, 1].set_xlabel("Material")
    axes[1, 1].set_ylabel("Strength (MPa)")

    fig.suptitle("Statistical Summary Dashboard", fontsize=18)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def explore_dataset(name: str, data: pd.DataFrame) -> None:
    """Print dataset shape, type info, head, and missing value summary."""
    print(f"--- {name} Dataset Overview ---")
    print(f"Shape: {data.shape}")

    buffer = StringIO()
    data.info(buf=buffer)
    print(buffer.getvalue())

    missing_summary = data.isna().sum()
    if missing_summary.any():
        print("Missing values per column:\n", missing_summary)
    else:
        print("No missing values detected.")

    print("Preview (first 5 rows):")
    print(data.head())
    print()


def main() -> None:
    """Execute the full Lab 4 analysis workflow."""
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

    concrete_data = load_data("concrete_strength.csv")
    material_data = load_data("material_properties.csv").rename(
        columns={"material_type": "material", "yield_strength_mpa": "strength_mpa"}
    )
    structural_loads = load_data("structural_loads.csv")

    explore_dataset("Concrete Strength", concrete_data)
    explore_dataset("Material Properties", material_data)
    explore_dataset("Structural Loads", structural_loads)

    concrete_stats = calculate_descriptive_stats(concrete_data, column="strength_mpa")
    structural_stats = structural_loads["load_kN"].describe().rename("structural_load_kN")

    print("Concrete Strength Descriptive Statistics:\n")
    print(concrete_stats.round(4))
    print("\n")
    print("Structural Load Summary Statistics:\n")
    print(structural_stats.round(4))
    print("\n")

    plot_distribution(
        concrete_data,
        column="strength_mpa",
        title="Concrete Strength Distribution with Central Tendencies",
        save_path=OUTPUT_DIR / "concrete_strength_distribution.png",
    )
    plot_spread_bands(
        concrete_data,
        column="strength_mpa",
        title="Concrete Strength with Standard Deviation Bands",
        save_path=OUTPUT_DIR / "concrete_strength_spread.png",
    )
    plot_distribution(
        structural_loads,
        column="load_kN",
        title="Structural Load Distribution",
        save_path=OUTPUT_DIR / "structural_load_distribution.png",
    )
    plot_material_comparison(
        material_data,
        column="strength_mpa",
        group_column="material",
        title="Material Strength Comparison",
        save_path=OUTPUT_DIR / "material_comparison_boxplot.png",
    )
    material_summary = (
        material_data.groupby("material")["strength_mpa"]
        .agg(count="count", mean="mean", std="std")
        .reset_index()
    )

    print("Material Strength Summary Statistics:\n")
    print(material_summary.set_index("material")[["count", "mean", "std"]].round(4))
    print("\n")

    create_statistical_summary_dashboard(
        concrete_data,
        material_data,
        material_summary=material_summary,
        save_path=OUTPUT_DIR / "statistical_summary_dashboard.png",
    )

    fitted_mu_sigma = fit_distribution(concrete_data, column="strength_mpa", distribution_type="normal")
    plot_distribution_fitting(
        concrete_data,
        column="strength_mpa",
        fitted_params=fitted_mu_sigma,
        save_path=OUTPUT_DIR / "distribution_fitting.png",
    )

    _, distribution_stats = create_probability_distributions_plot(
        save_path=OUTPUT_DIR / "probability_distributions.png"
    )

    probability_results = {
        "Bernoulli mean [p=0.95]": BERNOULLI_PASS_PROB,
        "Bernoulli variance [p=0.95]": BERNOULLI_PASS_PROB * (1 - BERNOULLI_PASS_PROB),
        "Binomial P(X=3) [n=100, p=0.05]": calculate_probability_binomial(BINOMIAL_TRIALS, BINOMIAL_DEFECT_RATE, 3),
        "Binomial P(X<=5) [n=100, p=0.05]": float(binom.cdf(5, BINOMIAL_TRIALS, BINOMIAL_DEFECT_RATE)),
        "Poisson P(X=8) [lambda=10]": calculate_probability_poisson(POISSON_EVENTS_PER_INTERVAL, 8),
        "Poisson P(X>15) [lambda=10]": 1 - poisson.cdf(15, POISSON_EVENTS_PER_INTERVAL),
        "Normal P(X>280) [mu=250, sigma=15]": calculate_probability_normal(NORMAL_MEAN, NORMAL_STD, x_lower=280),
        "Normal 95th percentile [mu=250, sigma=15]": float(norm.ppf(0.95, loc=NORMAL_MEAN, scale=NORMAL_STD)),
        "Exponential P(X<500) [mean=1000]": calculate_probability_exponential(EXPONENTIAL_MEAN, 500),
        "Exponential P(X>1500) [mean=1000]": calculate_probability_exponential(EXPONENTIAL_MEAN, 1500, survival=True),
    }

    for description, value in probability_results.items():
        print(f"{description}: {value:.4f}")
    print("\n")

    print("Distribution Sample vs. Theoretical Statistics:\n")
    for name, stats_dict in distribution_stats.items():
        sample_mean = stats_dict["sample_mean"]
        sample_var = stats_dict["sample_variance"]
        theoretical_mean = stats_dict["theoretical_mean"]
        theoretical_var = stats_dict["theoretical_variance"]
        print(f"{name}:")
        print(f"  Sample mean: {sample_mean:.4f}")
        print(f"  Theoretical mean: {theoretical_mean:.4f}")
        print(f"  Sample variance: {sample_var:.4f}")
        print(f"  Theoretical variance: {theoretical_var:.4f}\n")

    prior_damage = 0.05
    sensitivity = 0.95
    specificity = 0.90
    posterior = apply_bayes_theorem(prior_damage, sensitivity, specificity)
    print(f"Posterior Probability of Damage given Positive Test: {posterior:.4f}\n")

    plot_probability_tree(
        prior_damage,
        sensitivity,
        specificity,
        save_path=OUTPUT_DIR / "probability_tree.png",
    )

    report_path = create_statistical_report(
        concrete_stats=concrete_stats,
        structural_stats=structural_stats,
        probability_results=probability_results,
        distribution_stats=distribution_stats,
        material_summary=material_summary,
        posterior_probability=posterior,
        output_file="lab4_statistical_report.txt",
    )
    print(f"Report generated at: {report_path}")


if __name__ == "__main__":
    main()

