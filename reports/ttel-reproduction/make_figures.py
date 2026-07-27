#!/usr/bin/env python3
"""Render the reader-facing figures from the compact evidence summary."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
DATA = json.loads((HERE.parents[1] / "results" / "summary.json").read_text())
OUT = HERE / "images"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "figure.dpi": 160,
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
    }
)
COLORS = {
    "Independent": "#6b7280",
    "Refinement": "#e07a5f",
    "TTEL": "#2563eb",
    "No null": "#f59e0b",
    "Environment": "#059669",
}


def save(name: str) -> None:
    plt.tight_layout()
    plt.savefig(OUT / name, bbox_inches="tight")
    plt.close()


expanded = DATA["expanded"]

# Figure 1: the central pass-versus-generated-token frontier.
plt.figure(figsize=(7.2, 4.5))
for key, label in (
    ("independent", "Independent"),
    ("refinement", "Refinement"),
    ("filtered_ttel", "TTEL"),
):
    curve = np.asarray(expanded[key]["curve"])
    plt.plot(
        curve[:, 0],
        100 * curve[:, 1],
        marker="o",
        linewidth=2.5,
        color=COLORS[label],
        label=label,
    )
for x, y in np.asarray(expanded["filtered_ttel"]["curve"]):
    plt.annotate(f"{100*y:.1f}%", (x, 100*y), xytext=(5, 5), textcoords="offset points")
plt.xlabel("Mean generated tokens per task")
plt.ylabel("Tasks solved by budget (%)")
plt.title("Prefix branching moves the pass–token frontier")
plt.legend(frameon=False)
plt.grid(axis="both", alpha=0.18)
save("headline_frontier.png")

# Figure 2: null-feedback subtraction controls the number and location of spikes.
metrics = ["Qualifying spikes", "Branch position (% trace)"]
filtered_values = [
    expanded["filtered_ttel"]["mean_spikes"],
    100 * expanded["filtered_ttel"]["mean_branch_fraction"],
]
no_null_values = [
    expanded["no_null_ttel"]["mean_spikes"],
    100 * expanded["no_null_ttel"]["mean_branch_fraction"],
]
x = np.arange(2)
width = 0.34
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.8))
for ax, idx in zip(axes, range(2)):
    ax.bar(-width / 2, filtered_values[idx], width, color=COLORS["TTEL"], label="With null filter")
    ax.bar(width / 2, no_null_values[idx], width, color=COLORS["No null"], label="Without null filter")
    ax.set_xticks([])
    ax.set_title(metrics[idx])
    ax.grid(axis="y", alpha=0.18)
    for xpos, value in ((-width / 2, filtered_values[idx]), (width / 2, no_null_values[idx])):
        ax.text(xpos, value, f"{value:.1f}", ha="center", va="bottom")
axes[0].set_ylabel("Mean per failed trajectory")
axes[1].set_ylabel("Mean branch location")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
fig.suptitle("Null-feedback subtraction removes late, nonspecific branch candidates", fontweight="bold")
fig.tight_layout(rect=(0, 0.12, 1, 0.92))
fig.savefig(OUT / "null_filter_localization.png", bbox_inches="tight")
plt.close(fig)

# Figure 3: downstream result of the null ablation under both feedback types.
labels = ["Generic\nfiltered", "Generic\nno null", "Env.\nfiltered", "Env.\nno null"]
passes = [
    100 * expanded["filtered_ttel"]["curve"][-1][1],
    100 * expanded["no_null_ttel"]["curve"][-1][1],
    100 * expanded["environment_ttel"]["curve"][-1][1],
    100 * expanded["environment_no_null_ttel"]["curve"][-1][1],
]
tokens = [
    expanded["filtered_ttel"]["curve"][-1][0],
    expanded["no_null_ttel"]["curve"][-1][0],
    expanded["environment_ttel"]["curve"][-1][0],
    expanded["environment_no_null_ttel"]["curve"][-1][0],
]
downstream_colors = [COLORS["TTEL"], COLORS["No null"], COLORS["Environment"], "#a16207"]
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.8))
axes[0].bar(labels, passes, color=downstream_colors)
axes[0].set_ylabel("Pass@4 (%)")
axes[0].set_ylim(0, max(passes) * 1.25)
axes[1].bar(labels, tokens, color=downstream_colors)
axes[1].set_ylabel("Mean generated tokens")
axes[1].set_ylim(0, max(tokens) * 1.25)
for ax, vals, fmt in ((axes[0], passes, "{:.1f}%"), (axes[1], tokens, "{:,.0f}")):
    ax.tick_params(axis="x", rotation=0, labelsize=8)
    ax.grid(axis="y", alpha=0.18)
    for i, value in enumerate(vals):
        ax.text(i, value, fmt.format(value), ha="center", va="bottom")
fig.suptitle("The null filter helped downstream only with richer feedback", fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.92))
fig.savefig(OUT / "null_filter_downstream.png", bbox_inches="tight")
plt.close(fig)

# Figure 4: matched generic versus environment feedback.
feedback_names = ["Generic feedback", "Environment feedback"]
feedback_pass = [
    100 * expanded["filtered_ttel"]["curve"][-1][1],
    100 * expanded["environment_ttel"]["curve"][-1][1],
]
feedback_tokens = [
    expanded["filtered_ttel"]["curve"][-1][0],
    expanded["environment_ttel"]["curve"][-1][0],
]
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.8))
axes[0].bar(feedback_names, feedback_pass, color=[COLORS["TTEL"], COLORS["Environment"]])
axes[0].set_ylabel("Pass@4 (%)")
axes[0].set_ylim(0, max(feedback_pass) * 1.25)
axes[1].bar(feedback_names, feedback_tokens, color=[COLORS["TTEL"], COLORS["Environment"]])
axes[1].set_ylabel("Mean generated tokens")
axes[1].set_ylim(0, max(feedback_tokens) * 1.25)
for ax, vals, fmt in (
    (axes[0], feedback_pass, "{:.1f}%"),
    (axes[1], feedback_tokens, "{:,.0f}"),
):
    ax.tick_params(axis="x", rotation=10)
    ax.grid(axis="y", alpha=0.18)
    for i, value in enumerate(vals):
        ax.text(i, value, fmt.format(value), ha="center", va="bottom")
fig.suptitle("Executable feedback saves tokens on the expanded slice", fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.92))
fig.savefig(OUT / "environment_feedback.png", bbox_inches="tight")
plt.close(fig)

# Figure 5: expanded branch rule and pilot threshold sensitivities.
pilot = DATA["pilot"]
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.8))
branch_names = ["Raw token", "Syntax-aware"]
branch_pass = [
    100 * expanded["filtered_ttel"]["curve"][-1][1],
    100 * expanded["syntax_ttel"]["curve"][-1][1],
]
axes[0].bar(branch_names, branch_pass, color=[COLORS["TTEL"], "#8b5cf6"])
axes[0].set(title="Branch rule (24 tasks)", ylabel="Pass@4 (%)")
threshold_names = ["τ=0.04", "τ=0.06", "τ=0.10"]
threshold_pass = [
    100 * pilot["tau_0_04"]["pass_rate"],
    100 * pilot["filtered_ttel"]["pass_rate"],
    100 * pilot["tau_0_10"]["pass_rate"],
]
axes[1].bar(threshold_names, threshold_pass, color=["#93c5fd", COLORS["TTEL"], "#1d4ed8"])
axes[1].set(title="Spike threshold (12-task pilot)", ylabel="Pass@4 (%)")
for ax, vals in ((axes[0], branch_pass), (axes[1], threshold_pass)):
    ax.set_ylim(0, max(vals) * 1.25)
    ax.grid(axis="y", alpha=0.18)
    for i, value in enumerate(vals):
        ax.text(i, value, f"{value:.1f}%", ha="center", va="bottom")
fig.suptitle("Raw-token branching and τ=0.06 were the strongest settings", fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.92))
fig.savefig(OUT / "robustness_sensitivity.png", bbox_inches="tight")
plt.close(fig)
