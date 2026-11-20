"""Read the runs aligned at the site the map chose, and name the outcome. No GPU.

context.md §8.5's second step moves the alignment to where the baseline's answer depends
most, and asks two things of the runs trained there: does the answer lean on the new site
less than the baseline's does, and is that the labels' doing or the loss's. What each
outcome means was written down before the runs existed, as a table; this computes the
four paired differences the table is read from and names the row they fall in, so
choosing the sentence is not a judgement left for the day the numbers arrive.

The relative sensitivity comes from `probe_dependence_map.py` (its `reported` and
`position` sets), read at the chosen site — the mean over its positions when the choice
was all six — and at the width fixed beforehand, with the other widths beside it. The
accuracy is the official evaluator's. The baseline is the reported one throughout: it has
no alignment to move, so it was not retrained.

The separation the alignment reached at the new site is reported too, as the check that
the loss took hold there at all (§4.4) — without it, a sensitivity that did not fall
could mean the structure was used or that it was never built.
"""

from __future__ import annotations

from statistics import fmean
from typing import Any

from taskalign.analysis import (
    DIMS,
    PRIMARY_DIM,
    RELOCATED,
    RELOCATED_ARMS,
    SEEDS,
    PairedDifference,
    analysis_path,
    constrained_positions,
    eval_report,
    finished_timestamp,
    paired_difference,
    reported_accuracy,
    site_sensitivity,
    summarize,
)
from taskalign.utils import load_json, save_json

MAP_PATH = analysis_path("probe_dependence_map.py")
OUT_PATH = analysis_path(__file__)

# The rows of context.md §8.5's second-step table, in its order.
ROWS = (
    "falls_by_labels",
    "falls_not_by_labels",
    "kept_accuracy_up",
    "kept_accuracy_down",
    "kept_accuracy_undecided",
)


def falls(d: PairedDifference) -> bool:
    """The interval excludes zero and the difference is negative."""
    return d["excludes_zero"] and d["mean"] < 0


def rises(d: PairedDifference) -> bool:
    """The interval excludes zero and the difference is positive."""
    return d["excludes_zero"] and d["mean"] > 0


def row(sensitivity: dict[str, PairedDifference], accuracy: dict[str, PairedDifference]) -> str:
    """Which row of the table the four differences fall in."""
    if falls(sensitivity["proposal_minus_baseline"]):
        by_labels = falls(sensitivity["proposal_minus_ablation"])
        return "falls_by_labels" if by_labels else "falls_not_by_labels"
    if all(rises(d) for d in accuracy.values()):
        return "kept_accuracy_up"
    if falls(accuracy["proposal_minus_baseline"]):
        return "kept_accuracy_down"
    return "kept_accuracy_undecided"


def compared(by_arm: dict[str, dict[int, float]]) -> dict[str, Any]:
    """Each arm across the seeds, and the two differences the table reads."""
    differences = {
        "proposal_minus_baseline": paired_difference(by_arm["proposal"], by_arm["baseline"]),
        "proposal_minus_ablation": paired_difference(by_arm["proposal"], by_arm["ablation"]),
    }
    return {**{arm: summarize(values) for arm, values in by_arm.items()}, **differences}


def separation(report: dict[str, Any]) -> float:
    """A run's Cohen's d between same- and different-signature pairs, averaged over the
    sites it aligned — the report nests by site only when there is more than one.
    """
    stability = report["representation_stability"]
    if "separation_d" in stability:
        return stability["separation_d"]
    return fmean(site["separation_d"] for site in stability.values())


if __name__ == "__main__":
    mapped = load_json(MAP_PATH)
    sets, choice = mapped["sets"], mapped["choice"]
    if "position" not in sets:
        raise SystemExit(
            f"{MAP_PATH.name} holds no runs aligned at the chosen site. Train them "
            f"(`sweep.py position`), then measure them (`probe_dependence_map.py position`)."
        )
    layer, positions = choice["layer"], tuple(choice["positions"])

    stamps = {
        arm: {seed: finished_timestamp(RELOCATED, arm, seed) for seed in SEEDS}
        for arm in RELOCATED_ARMS
    }
    for arm, by_seed in stamps.items():
        for seed, stamp in by_seed.items():
            if stamp is None:
                raise SystemExit(f"{RELOCATED} {arm} seed {seed} has not finished training.")
            if (trained := constrained_positions(arm, seed, RELOCATED, stamp)) != positions:
                raise SystemExit(
                    f"{RELOCATED} {arm} seed {seed} aligned positions {trained}, but the "
                    f"map now chooses {positions}. The runs and the map are not one experiment."
                )

    def at_site(arm: str, seed: int, m: int) -> float:
        runs = (sets["reported"]["baseline"] if arm == "baseline"
                else sets["position"][f"{RELOCATED}_{arm}"])
        return site_sensitivity(runs[str(seed)], layer, positions, m)

    sensitivity = {
        str(m): compared({
            arm: {seed: at_site(arm, seed, m) for seed in SEEDS}
            for arm in ("baseline", *RELOCATED_ARMS)
        })
        for m in DIMS
    }
    accuracy = compared({
        "baseline": {seed: reported_accuracy("baseline", seed) for seed in SEEDS},
        **{
            arm: {seed: reported_accuracy(arm, seed, RELOCATED, stamps[arm][seed])
                  for seed in SEEDS}
            for arm in RELOCATED_ARMS
        },
    })
    separations = {
        arm: summarize({
            seed: separation(eval_report(arm, seed, RELOCATED, stamps[arm][seed]))
            for seed in SEEDS
        })
        for arm in RELOCATED_ARMS
    }

    differences = ("proposal_minus_baseline", "proposal_minus_ablation")
    at_primary = sensitivity[str(PRIMARY_DIM)]
    outcome = row(
        {name: at_primary[name] for name in differences},
        {name: accuracy[name] for name in differences},
    )

    def bracket(d: PairedDifference, unit: str = "") -> str:
        star = " *" if d["excludes_zero"] else ""
        return f"{d['mean']:+.4f}{unit} [{d['ci_low']:+.4f}, {d['ci_high']:+.4f}]{star}"

    print(f"aligned at layer {layer}, positions {list(positions)} "
          f"(largest baseline sensitivity at {choice['largest']})\n")
    for m in DIMS:
        read = sensitivity[str(m)]
        mark = "  ← the width the verdict is read at" if m == PRIMARY_DIM else ""
        print(f"relative sensitivity, m = {m}{mark}")
        for arm in ("baseline", *RELOCATED_ARMS):
            print(f"{arm:>26}  {read[arm]['mean']:.4f}")
        for name in differences:
            print(f"{name:>26}  {bracket(read[name])}")
    print("\naccuracy (%)")
    for arm in ("baseline", *RELOCATED_ARMS):
        print(f"{arm:>26}  {accuracy[arm]['mean']:.2f}")
    for name in differences:
        print(f"{name:>26}  {bracket(accuracy[name], 'pt')}")
    print("\nseparation d at the aligned site: "
          + ", ".join(f"{arm} {separations[arm]['mean']:.2f}" for arm in RELOCATED_ARMS))
    print(f"\n* interval excludes 0.\nrow {ROWS.index(outcome) + 1} of §8.5's table: {outcome}")

    save_json({
        "layer": layer,
        "positions": list(positions),
        "largest": choice["largest"],
        "primary_dim": PRIMARY_DIM,
        "timestamps": {arm: {str(s): t for s, t in by.items()} for arm, by in stamps.items()},
        "sensitivity": sensitivity,
        "accuracy": accuracy,
        "separation_d": separations,
        "row": ROWS.index(outcome) + 1,
        "outcome": outcome,
    }, OUT_PATH)
    print(f"\nwritten: {OUT_PATH}")
