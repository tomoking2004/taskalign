"""The trade-off the study states at one point, read as a curve. No GPU.

The reported experiment fixes the alignment weight at 0.1 and concludes that the
signature becomes far more readable while the answers cost almost nothing. Both halves of
that sentence are single measurements, and a single measurement cannot say whether the
readability is bought cheaply or the cost simply had not started yet.

This puts the two quantities side by side at every setting that has been run — a weight
or a layer: what the representation carries, measured the way §5.1 measures it, against
what the official evaluator scores. Weight 0 and weight 0.1 come from the reported
experiment's baseline and proposal runs; the rest come from `sweep.py`, and are found by
their variant names — a variant that has not finished training is simply not a point yet.

Where `probe_dependence_map.py` has measured a point, its relative sensitivity at the
aligned site stands beside the other two (context.md §8.4): whether the answer's
dependence on the site falls as the weight rises is the third thing the curve is for. The
weights are then read as one curve at the sweep's seed — the reported points at that same
seed, not at their three-seed mean, so every point on it differs by its weight alone.

Runs at one seed carry no interval, and none is printed for them — the shape of the curve
is what one seed is for, and an interval drawn through single points would suggest a
precision that was not paid for.
"""

from __future__ import annotations

from itertools import pairwise
from typing import Any

import torch.nn.functional as F

from taskalign.analysis import (
    DIMS,
    LAYER_VARIANTS,
    PRIMARY_DIM,
    SEEDS,
    STRENGTH_VARIANTS,
    SWEEP_SEEDS,
    analysis_path,
    batch_of,
    constrained_layer,
    finished_timestamp,
    keep_frequent,
    paired_difference,
    probe_accuracy,
    reported_accuracy,
    representations,
    run_config,
    scorable,
    site_sensitivity,
    split_mask,
    splitter,
    summarize,
)
from taskalign.config import cfg
from taskalign.utils import load_json, save_json

# The arm every swept variant ran, and the two reported points on the same axis: the
# reported baseline is the alignment weight 0 point and the reported proposal the 0.1 one.
# The weight itself is never written here — it is read from each run's own config, so a
# point cannot be filed under a weight it was not trained with.
ARM      = "proposal"
REPORTED = ("baseline", "proposal")

MAP_PATH = analysis_path("probe_dependence_map.py")
OUT_PATH = analysis_path(__file__)


def spread_or_value(by_seed: dict[int, float]) -> dict[str, Any]:
    """`summarize` where there are seeds to spread over; the bare value where there is one.

    A swept point runs at a single seed, and a spread over one number would be an
    invention — the mean *is* the measurement, and `per_seed` says which seed it was.
    """
    if len(by_seed) > 1:
        return summarize(by_seed)
    return {"mean": next(iter(by_seed.values())),
            "per_seed": {str(seed): value for seed, value in by_seed.items()}}


def point(variant: str, arm: str, stamps: dict[int, str | None]) -> dict[str, Any]:
    """One configuration: what it was trained with, and what it came to.

    The probe is fitted per seed on that seed's representations, exactly as §5.1 fits it,
    and the split is drawn from the same fixed source — so a number here and a number
    there differ by the run and by nothing else. `stamps` names each seed's run; `None`
    is the reported experiment's own batch.
    """
    weights = {run_config(arm, seed, variant, ts)["align_weight"] for seed, ts in stamps.items()}
    if len(weights) != 1:
        raise SystemExit(
            f"{variant or 'reported'} {arm}: seeds disagree about the alignment weight "
            f"({sorted(weights)}), so they are not one point on the curve."
        )

    probe, accuracy = {}, {}
    for seed, ts in stamps.items():
        features, signatures, _ = representations(arm, seed, variant, ts)
        keep, _, labels = keep_frequent(signatures)
        rows, sub_labels, sub_train, n_classes = scorable(
            labels, split_mask(len(keep), splitter())
        )
        x = F.normalize(features[keep][rows].float(), dim=-1)
        probe[seed] = probe_accuracy(x, sub_labels, sub_train, n_classes)
        accuracy[seed] = reported_accuracy(arm, seed, variant, ts)

    seed, ts = next(iter(stamps.items()))
    return {
        "arm": arm,
        "variant": variant,
        "align_weight": weights.pop(),
        "layer": constrained_layer(arm, seed, variant, ts)[0],
        "seeds": list(stamps),
        "timestamps": {str(seed): ts or batch_of(seed) for seed, ts in stamps.items()},
        "probe_accuracy": spread_or_value(probe),
        "accuracy": spread_or_value(accuracy),
        "_probe_per_seed": probe,
        "_accuracy_per_seed": accuracy,
    }


def sensitivities(p: dict[str, Any], mapped: dict[str, Any]) -> dict[str, dict[str, float]]:
    """A point's relative sensitivity at its own aligned site, by width and by seed —
    empty where the map has not measured the point.
    """
    key = f"{p['variant']}_{p['arm']}" if p["variant"] else p["arm"]
    for runs in mapped.get("sets", {}).values():
        if key in runs:
            return {
                str(m): {seed: site_sensitivity(run, p["layer"], (-1,), m)
                         for seed, run in runs[key].items()}
                for m in DIMS
            }
    return {}


def strength_curve(points: list[dict[str, Any]], seed: int) -> dict[str, Any] | None:
    """The points that differ by weight alone, at one seed, in order of weight.

    `None` until every weight has trained: a curve with a point missing is read as a
    shape it may not have. The sensitivity joins once the map has measured every point,
    and with it the one thing asked of its shape (context.md §8.4) — whether it falls
    at every step as the weight rises.
    """
    names = ("", *STRENGTH_VARIANTS)
    curve = sorted((p for p in points if p["variant"] in names), key=lambda p: p["align_weight"])
    if len(curve) != len(REPORTED) + len(STRENGTH_VARIANTS):
        return None
    read: dict[str, Any] = {
        "seed": seed,
        "align_weight": [p["align_weight"] for p in curve],
        "probe_accuracy": [p["_probe_per_seed"][seed] for p in curve],
        "accuracy": [p["_accuracy_per_seed"][seed] for p in curve],
    }
    if all(p["sensitivity"] for p in curve):
        read["sensitivity"] = {
            str(m): [p["sensitivity"][str(m)][str(seed)] for p in curve] for m in DIMS
        }
        primary = read["sensitivity"][str(PRIMARY_DIM)]
        read["sensitivity_falls_at_every_step"] = all(
            after < before for before, after in pairwise(primary)
        )
    return read


if __name__ == "__main__":
    if not cfg.gqa.testdev_questions.exists():
        raise SystemExit(f"testdev questions not found at {cfg.gqa.testdev_questions}.")

    wanted: list[tuple[str, str, dict[int, str | None]]] = [
        ("", arm, dict.fromkeys(SEEDS)) for arm in REPORTED
    ]
    for variant in (*LAYER_VARIANTS, *STRENGTH_VARIANTS):
        stamps = {seed: finished_timestamp(variant, ARM, seed) for seed in SWEEP_SEEDS}
        if all(stamps.values()):
            wanted.append((variant, ARM, stamps))
        else:
            print(f"{variant}: not trained yet — left out")

    mapped = load_json(MAP_PATH) if MAP_PATH.exists() else {}
    points = [point(variant, arm, stamps) for variant, arm, stamps in wanted]
    for p in points:
        p["sensitivity"] = sensitivities(p, mapped)
    points.sort(key=lambda p: p["align_weight"])

    # Every point against the alignment-free one, which is the only reference the curve
    # has: "costs almost nothing" is a claim about the distance from weight 0.
    zero = next(p for p in points if p["align_weight"] == 0)

    print(f"\n{'weight':>8}  {'variant':>12}  {'probe':>8}  {'accuracy':>9}  "
          f"{'vs weight 0':>12}  {f'e (m={PRIMARY_DIM})':>10}  seeds")
    for p in points:
        against = ""
        if p is not zero:
            shared = sorted(set(p["_accuracy_per_seed"]) & set(zero["_accuracy_per_seed"]))
            if len(shared) >= 2:
                d = paired_difference(
                    {s: p["_accuracy_per_seed"][s] for s in shared},
                    {s: zero["_accuracy_per_seed"][s] for s in shared},
                )
                p["accuracy_vs_zero"] = d
                against = f"{d['mean']:+.2f} {'*' if d['excludes_zero'] else ''}"
            elif shared:
                delta = p["_accuracy_per_seed"][shared[0]] - zero["_accuracy_per_seed"][shared[0]]
                p["accuracy_vs_zero"] = {"mean": delta, "seeds": shared}
                against = f"{delta:+.2f}"
        at_site = p["sensitivity"].get(str(PRIMARY_DIM), {})
        leaning = f"{sum(at_site.values()) / len(at_site):.4f}" if at_site else "—"
        print(f"{p['align_weight']:>8}  {p['variant'] or p['arm']:>12}  "
              f"{p['probe_accuracy']['mean']:>7.1%}  {p['accuracy']['mean']:>9.2f}  "
              f"{against:>12}  {leaning:>10}  {','.join(map(str, p['seeds']))}")

    if len(points) > 2:
        best = max(points, key=lambda p: p["probe_accuracy"]["mean"])
        print(f"\nprobe accuracy peaks at {best['variant'] or best['arm']}, weight "
              f"{best['align_weight']} ({best['probe_accuracy']['mean']:.1%}), "
              f"{100 * (best['probe_accuracy']['mean'] - zero['probe_accuracy']['mean']):+.1f} "
              f"pt over no alignment")
    print("\n* interval excludes 0. Points with one seed carry no interval.")

    curve = strength_curve(points, SWEEP_SEEDS[0])
    if curve and "sensitivity" in curve:
        falls = curve["sensitivity_falls_at_every_step"]
        print(f"\nseed {curve['seed']}, weights {curve['align_weight']}: relative sensitivity "
              f"at m = {PRIMARY_DIM} " + " → ".join(
                  f"{v:.4f}" for v in curve["sensitivity"][str(PRIMARY_DIM)])
              + (" — falls at every step" if falls else " — does not fall at every step"))

    save_json({
        "points": [{k: v for k, v in p.items() if not k.startswith("_")} for p in points],
        "strength_curve": curve,
    }, OUT_PATH)
    print(f"\nwritten: {OUT_PATH}")
