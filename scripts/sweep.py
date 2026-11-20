"""Vary one setting the reported experiment fixed, so a single point becomes a curve.

The study measured one configuration: alignment weight 0.1, on layer 16, at the last
prompt token. Everything it concludes is therefore about that point. This runs the
neighbouring ones, one setting at a time:

    python scripts/sweep.py strength    alignment weight 0.03, 0.3 and 1.0 (context.md §8.4)
    python scripts/sweep.py layer       layer 8 and layer 24 (§A.5)
    python scripts/sweep.py position    the position the map chose (§8.5, second step)

Never two at once — a curve mixing two changed settings has two explanations and settles
neither.

`strength` and `layer` ask what the shape of a curve is, so they run the signature-labelled
arm at one seed: the shape is visible at one seed, and three seeds at every point costs
three times the GPU hours to sharpen intervals nobody reads until the shape says where to
look. Two points are already measured and are deliberately absent: weight 0 *is* the
baseline arm, and weight 0.1 *is* the reported proposal.

`position` asks for a verdict instead — whether the answer leans on the new site less,
and whether that is the labels' doing — so it runs both aligned arms at the reported
seeds. The baseline is absent here too: it has no alignment to move, so the reported
baseline is the baseline at every position. Where to align is not written in this file.
`probe_dependence_map.py reported` chose it from the baseline's map by a rule fixed
beforehand, and this trains what that wrote.

The runs land under their own timestamp, and are still comparable to the reported nine:
a seed fixes the initialisation and the batch order, so a run here at seed 42 sees
exactly what the reported runs at seed 42 saw. Nothing needs the timestamp afterwards —
the analyses find a variant's finished run by its name — and a run that already finished
is not trained again, so a sweep that was interrupted resumes where it stopped.

Start with `--debug`. It finishes in minutes and proves the wiring before a day of GPU
time is committed to it. Remove the `*_debug` runs it leaves in runs/ and snapshots/.
"""

from __future__ import annotations

import os

# Must be set before the CUDA allocator is first used (i.e. before .to("cuda")).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
from collections.abc import Callable
from itertools import product
from typing import NamedTuple

from train4all.utils import empty_cuda_cache

from taskalign.analysis import (
    LAYER_VARIANTS,
    RELOCATED,
    RELOCATED_ARMS,
    SEEDS,
    STRENGTH_VARIANTS,
    SWEEP_SEEDS,
    Arm,
    analysis_path,
    finished_timestamp,
)
from taskalign.training import Settings, run_training
from taskalign.utils import load_json, timestamp

MAP_PATH = analysis_path("probe_dependence_map.py")

Variants = tuple[tuple[str, Settings], ...]


def strengths() -> Variants:
    """The alignment weights either side of the reported 0.1."""
    weights = (0.03, 0.3, 1.0)
    return tuple(zip(
        STRENGTH_VARIANTS, (Settings(align_weight=w) for w in weights), strict=True
    ))


def layers() -> Variants:
    """The other question the study leaves open: "why layer 16?" """
    return tuple(zip(LAYER_VARIANTS, (Settings(layers=(8,)), Settings(layers=(24,))), strict=True))


def chosen_position() -> Variants:
    """The site the baseline's map chose, at the layer and weight the study reports."""
    if not MAP_PATH.exists():
        raise SystemExit(
            f"{MAP_PATH.name} is not there, so no position has been chosen. "
            f"Run `python scripts/probe_dependence_map.py reported` first."
        )
    choice = load_json(MAP_PATH)["choice"]
    site = Settings(layers=(choice["layer"],), positions=tuple(choice["positions"]))
    return ((RELOCATED, site),)


class Sweep(NamedTuple):
    """What one sweep changes, and the arms and seeds it runs each change at."""

    variants: Callable[[], Variants]
    arms: tuple[Arm, ...]
    seeds: tuple[int, ...]


# The variant names are the analyses' (`taskalign.analysis.experiment`): a name becomes
# part of the run directory, and it is what the analyses find the run by.
SWEEPS: dict[str, Sweep] = {
    "strength": Sweep(strengths, ("proposal",), SWEEP_SEEDS),
    "layer":    Sweep(layers, ("proposal",), SWEEP_SEEDS),
    "position": Sweep(chosen_position, RELOCATED_ARMS, SEEDS),
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("sweep", choices=SWEEPS)
    parser.add_argument("--debug", action="store_true",
                        help="shrink every split to a smoke test that finishes in minutes")
    args = parser.parse_args()

    sweep = SWEEPS[args.sweep]
    ts = timestamp()
    runs = list(product(sweep.seeds, sweep.variants(), sweep.arms))
    print(f"sweep timestamp: {ts}\n")

    # Seed-major, so an interrupted sweep leaves whole comparisons behind.
    for i, (seed, (variant, settings), arm) in enumerate(runs, start=1):
        header = (f"Run {i}/{len(runs)}: {variant} {arm} (seed {seed})  ·  weight "
                  f"{settings.align_weight}, layers {settings.layers}, "
                  f"positions {settings.positions}")
        if not args.debug and (done := finished_timestamp(variant, arm, seed)):
            print(f"\n⏭️  {header}\n    already finished under {done} — not trained again")
            continue
        print(f"\n{'=' * 79}\n▶️  {header}\n{'=' * 79}")
        run_training(
            arm, seed, settings.as_debug() if args.debug else settings,
            timestamp=ts, variant=variant,
        )

        # Here rather than inside `run_training`: the model, its optimizer state and the
        # trainer holding both are still live locals until that call returns.
        empty_cuda_cache()

    print(f"\n{'=' * 79}\n✅ All {len(runs)} runs of the {args.sweep} sweep are in.\n{'=' * 79}\n")
