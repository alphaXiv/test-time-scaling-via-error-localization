# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "marimo>=0.14",
#   "matplotlib>=3.9",
# ]
# ///

import marimo

__generated_with = "0.23.15"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt

    return mo, plt


@app.cell
def _(mo):
    mo.md(r"""
    # Error localization as a test-time scaling strategy

    A failed program need not be thrown away. *Test-Time Scaling via Error
    Localization* proposes comparing token probabilities with and without
    failure feedback, keeping the prefix before the most diagnostic token,
    and sampling a new suffix. This notebook walks through our fresh,
    executable reproduction of that mechanism.

    **Verdict: partially reproduced.** On 24 LiveCodeBench V6 tasks and four
    seeds, filtered TTEL reached **32.3% pass@4 at 5,460 mean generated
    tokens**, versus **4.2% at 5,976 tokens** for independent sampling. The
    null-feedback filter localized sharply; downstream gains appeared with
    executable feedback but not with a generic failure sentence.
    """)
    return


@app.cell
def _():
    # Embedded terminal measurements: readers never need cluster artifacts.
    curves = {
        "Independent": [
            (1528.10, 0.03125),
            (3016.10, 0.03125),
            (4504.10, 0.041667),
            (5976.10, 0.041667),
        ],
        "Refinement": [
            (1536.00, 0.020833),
            (2997.27, 0.093750),
            (4374.30, 0.166667),
            (5609.81, 0.291667),
        ],
        "Filtered TTEL": [
            (1534.52, 0.020833),
            (2993.01, 0.135417),
            (4296.46, 0.208333),
            (5460.40, 0.322917),
        ],
        "No-null TTEL": [
            (1536.00, 0.020833),
            (2996.08, 0.156250),
            (4252.42, 0.260417),
            (5337.65, 0.333333),
        ],
        "Environment TTEL": [
            (1536.00, 0.031250),
            (2961.20, 0.135417),
            (4218.68, 0.291667),
            (5273.98, 0.333333),
        ],
        "Environment no-null TTEL": [
            (1536.00, 0.031250),
            (2991.33, 0.104167),
            (4310.89, 0.229167),
            (5406.64, 0.322917),
        ],
        "Syntax-aware TTEL": [
            (1529.12, 0.031250),
            (2972.12, 0.125000),
            (4281.74, 0.218750),
            (5435.16, 0.260417),
        ],
    }
    diagnostics = {
        "Filtered TTEL": {"spikes": 13.1824, "branch_fraction": 0.152425},
        "No-null TTEL": {"spikes": 126.758, "branch_fraction": 0.445945},
    }
    pilot = {
        "Generic feedback": (0.333333, 5306.02),
        "Environment feedback": (0.354167, 5175.56),
        "Syntax-aware branch": (0.250000, 5331.83),
        "Threshold 0.04": (0.291667, 5335.46),
        "Threshold 0.10": (0.291667, 5250.23),
    }
    return curves, diagnostics, pilot


@app.cell
def _(curves, mo, plt):
    _frontier_fig, _frontier_ax = plt.subplots(figsize=(8, 4.6))
    _frontier_colors = {
        "Independent": "#6b7280",
        "Refinement": "#e07a5f",
        "Filtered TTEL": "#2563eb",
    }
    for _label in ("Independent", "Refinement", "Filtered TTEL"):
        _points = curves[_label]
        _frontier_ax.plot(
            [_p[0] for _p in _points],
            [100 * _p[1] for _p in _points],
            marker="o",
            linewidth=2.5,
            color=_frontier_colors[_label],
            label=_label,
        )
    _frontier_ax.set(
        title="The central result: tasks solved per generated token",
        xlabel="Mean generated tokens per task",
        ylabel="Solved by budget (%)",
    )
    _frontier_ax.grid(alpha=0.2)
    _frontier_ax.legend(frameon=False)
    mo.vstack(
        [
            mo.md(
                "**Read upward and leftward as better.** Each point permits one "
                "additional attempt; TTEL separates after the first failed attempt."
            ),
            _frontier_fig,
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## From feedback to a branch point

    For a failed trajectory \(y_1,\ldots,y_T\), we teacher-force those same
    tokens under three prompts:

    1. the original problem (the *student*);
    2. the problem, failed answer, and true feedback (the *teacher*);
    3. the same history with a neutral instruction (the *null baseline*).

    A token qualifies when true feedback lowers its probability by more than
    0.06 while neutral feedback does not. Among qualifying tokens, we branch
    where the teacher/null separation is greatest. Independent sampling
    discards the whole answer; refinement puts the whole failed code back
    into a new prompt; TTEL alone preserves the selected prefix.
    """)
    return


@app.cell
def _(diagnostics, mo, plt):
    _diag_labels = list(diagnostics)
    _spikes = [diagnostics[_key]["spikes"] for _key in _diag_labels]
    _locations = [
        100 * diagnostics[_key]["branch_fraction"] for _key in _diag_labels
    ]
    _diag_fig, _diag_axes = plt.subplots(1, 2, figsize=(8, 3.5))
    _diag_colors = ["#2563eb", "#f59e0b"]
    _diag_axes[0].bar(_diag_labels, _spikes, color=_diag_colors)
    _diag_axes[0].set(title="Qualifying spikes", ylabel="Mean per failed trajectory")
    _diag_axes[1].bar(_diag_labels, _locations, color=_diag_colors)
    _diag_axes[1].set(title="Branch location", ylabel="Mean position (% of trace)")
    for _diag_ax in _diag_axes:
        _diag_ax.tick_params(axis="x", rotation=12)
        _diag_ax.grid(axis="y", alpha=0.2)
    mo.vstack(
        [
            mo.md(
                "Without null subtraction, spike count rose **9.6×** and the "
                "branch moved from 15.2% to 44.6% of the trajectory."
            ),
            _diag_fig,
        ]
    )
    return


@app.cell
def _(curves, mo):
    filtered = curves["Filtered TTEL"][-1]
    no_null = curves["No-null TTEL"][-1]
    mo.md(
        f"""
        ## The decisive ablation has a mixed answer

        The diagnostic half reproduced cleanly, but with generic feedback the
        downstream degradation did not. Filtered TTEL ended at
        **{100 * filtered[1]:.1f}%** pass@4 and
        **{filtered[0]:,.0f}** tokens; no-null TTEL ended at
        **{100 * no_null[1]:.1f}%** and **{no_null[0]:,.0f}** tokens. On this
        slice, more specific localization did not translate into more solved
        programs. With executable feedback, the filter instead improved pass@4
        by 1.0 point and reduced tokens by 2.5%. The downstream claim is
        therefore *partially aligned and feedback-dependent* here.
        """
    )
    return


@app.cell
def _(curves, mo):
    _generic = curves["Filtered TTEL"][-1]
    _environment = curves["Environment TTEL"][-1]
    mo.md(
        f"""
        ## Richer feedback helped efficiency

        Replacing the generic failure sentence with the first public-test error
        raised pass@4 from **{100 * _generic[1]:.1f}%** to
        **{100 * _environment[1]:.1f}%** and reduced mean generated tokens from
        **{_generic[0]:,.0f}** to **{_environment[0]:,.0f}**. A paired
        task-bootstrap interval was wide for the pass difference, but remained
        below zero for token use (−394 to −34).
        """
    )
    return


@app.cell
def _(mo, pilot):
    selector = mo.ui.dropdown(
        options=list(pilot),
        value="Environment feedback",
        label="Pilot sensitivity",
    )
    selector
    return (selector,)


@app.cell
def _(mo, pilot, selector):
    passed, tokens = pilot[selector.value]
    mo.md(
        f"""
        **{selector.value}:** {100 * passed:.1f}% pass@4 at {tokens:,.0f} mean
        generated tokens. These are 12-task pilot measurements, useful as
        directional sensitivities rather than headline estimates.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Scope, compute, and what remains

    We used a deterministic stdin-only subset of 24 LiveCodeBench V6 tasks
    (8 per difficulty), Qwen3-4B-Thinking-2507, four seeds, four attempts,
    and at most 1,536 generated tokens per attempt. All formal runs executed
    on Kubernetes using NVIDIA RTX PRO 6000 Blackwell GPUs, with 16 GPUs
    occupied concurrently at peak.

    The paper instead evaluated 131 tasks, up to 64 attempts, a 16,384-token
    cap, and primarily Qwen3-8B. A full reproduction still needs that scale,
    additional seeds, and confidence intervals. The repository report
    records the exact commands, run lineage, and per-claim assessment. The
    fresh Kubernetes evidence window lasted 2.31 hours.
    """)
    return


if __name__ == "__main__":
    app.run()
