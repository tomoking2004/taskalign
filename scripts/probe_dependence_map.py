"""Map where the answer depends — every layer, every prompt position — and choose a site.

`probe_dependence.py` measured how hard the readout leans on the aligned site and found
that alignment loosened it (context.md §A.4). The question that leaves is whether the
structure would be used if it were built somewhere else, and the first thing that needs is
a map of where the answer depends at all (§8.5, first step). This measures the same
relative sensitivity at every block's output and every prompt position, from the same
directions, the same prompts and the same number of backward passes — one backward reaches
them all (`taskalign.analysis.dependence`).

Name the runs to measure:

    python scripts/probe_dependence_map.py reported     the nine reported runs
    python scripts/probe_dependence_map.py layer        the two runs aligned at layers 8 and 24
    python scripts/probe_dependence_map.py strength     the runs at other alignment weights
    python scripts/probe_dependence_map.py position     the runs aligned at the chosen site

Each set is measured whole and replaces only its own entry in the one file they share, so
the sets can be measured days apart, as their runs finish training. A set measured under
different prompts or widths from the ones on file is refused rather than filed beside
them.

`reported` also makes the choice the second step trains on: among the positions that have
read the whole question, the one the baseline depends on most at the aligned layer — or
all of them, if that one is already the aligned position (`choose_positions`). The rule and
the width it is read at were fixed before the map existed; this only applies them, and
`sweep.py position` trains whatever is written here.

The cell at the aligned site is `probe_dependence.py`'s number again, and the two are
compared on the way out: a map that did not reproduce it would be a different measurement.

Add `--prompts 8` to prove the wiring in minutes. A trial prints and writes nothing.
"""

from __future__ import annotations

import argparse
from statistics import fmean
from typing import Any, NamedTuple

from train4all.utils import empty_cuda_cache

from taskalign.analysis import (
    ARMS,
    BATCH,
    CHECKPOINT_NAME,
    DIMS,
    LAYER_VARIANTS,
    MODEL,
    PREFIXES,
    PRIMARY_DIM,
    PROMPTS,
    RELOCATED,
    RELOCATED_ARMS,
    SEEDS,
    STRENGTH_VARIANTS,
    SWEEP_SEEDS,
    Arm,
    Layout,
    analysis_path,
    answer_token_rows,
    checkpoint_path,
    choose_positions,
    constrained_layer,
    finished_timestamp,
    jacobian_batches,
    measure_map,
    output_basis,
    site,
    site_sensitivity,
    trained_model,
)
from taskalign.config import cfg
from taskalign.models import build_smolvlm
from taskalign.utils import load_json, save_json

OUT_PATH        = analysis_path(__file__)
DEPENDENCE_PATH = analysis_path("probe_dependence.py")


class Run(NamedTuple):
    """One run to measure. An empty variant is the reported experiment."""

    variant: str
    arm: Arm
    seed: int

    @property
    def name(self) -> str:
        """What the run's readings are filed under, beside its seed."""
        return f"{self.variant}_{self.arm}" if self.variant else self.arm


SETS: dict[str, tuple[Run, ...]] = {
    "reported":  tuple(Run("", arm, seed) for arm in ARMS for seed in SEEDS),
    "layer":     tuple(Run(v, "proposal", s) for v in LAYER_VARIANTS for s in SWEEP_SEEDS),
    "strength":  tuple(Run(v, "proposal", s) for v in STRENGTH_VARIANTS for s in SWEEP_SEEDS),
    "position":  tuple(Run(RELOCATED, arm, seed) for arm in RELOCATED_ARMS for seed in SEEDS),
}


def located(runs: tuple[Run, ...]) -> dict[Run, str | None]:
    """Each run's timestamp, after checking that every one of them can be measured.

    Fails before the first model is loaded: a set is compared as a whole, and a missing
    run found after the others were measured would cost the pass.
    """
    stamps: dict[Run, str | None] = {}
    missing: list[str] = []
    for run in runs:
        stamp = finished_timestamp(*run) if run.variant else None
        stamps[run] = stamp
        if run.variant and stamp is None:
            missing.append(f"  {run.variant} {run.arm} seed {run.seed}: no finished snapshot")
        elif not (path := checkpoint_path(run.arm, run.seed, run.variant, stamp)).exists():
            missing.append(f"  {path}")
    if missing:
        raise SystemExit(
            f"{len(missing)} of {len(runs)} runs cannot be measured:\n" + "\n".join(missing)
            + "\n\nA variant is trained by scripts/sweep.py, and the weights stay on the "
              "machine that trained them."
        )
    return stamps


def reproduces(runs: dict[str, dict[str, Any]], layer: int, prompts: int) -> float | None:
    """The largest gap between the map's aligned-site cell and `probe_dependence.py`'s
    number, over the reported runs and every width — `None` when that file is absent or
    was measured over a different number of prompts than the `prompts` measured here.
    """
    if not DEPENDENCE_PATH.exists():
        return None
    filed = load_json(DEPENDENCE_PATH)
    if filed["jacobian_prompts"] != prompts:
        return None
    return max(
        abs(site_sensitivity(runs[arm][seed], layer, (-1,), m)
            - filed["runs"][arm][seed]["by_dim"][str(m)]["sensitivity"])
        for arm in runs for seed in runs[arm] for m in DIMS
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("set", choices=SETS)
    parser.add_argument("--prompts", type=int, default=PROMPTS,
                        help="fewer than the full set makes this a trial that writes nothing")
    args = parser.parse_args()

    stamps = located(SETS[args.set])
    layer, _ = site()

    testdev = load_json(cfg.gqa.testdev_questions)
    answer_rows = answer_token_rows({entry["answer"] for entry in testdev.values()}, MODEL)
    _, processor = build_smolvlm("500M")
    layout = Layout(*processor.prompt_layout())
    batches = jacobian_batches(processor, args.prompts, BATCH)
    counted = sum(len(b["prompt_lens"]) for b in batches)

    print(f"prompt layout: {layout.head} tokens before the question, {layout.tail} after")
    print(f"relative sensitivity at every block and position, over {counted} prompts\n")

    runs: dict[str, dict[str, Any]] = {}
    for run, stamp in stamps.items():
        # The layer this run aligned, which for a layer variant is not the reported one.
        own, _ = constrained_layer(run.arm, run.seed, run.variant, stamp)
        vlm, trained = trained_model(run.arm, run.seed, (own,), cfg.device, run.variant, stamp)
        span = output_basis(trained.output_weight, answer_rows, max(DIMS))
        entry = measure_map(vlm, batches, span, layout, own)
        del vlm
        empty_cuda_cache()

        entry["timestamp"] = stamp
        runs.setdefault(run.name, {})[str(run.seed)] = entry
        print(f"[{run.name} seed{run.seed}]  layer {own}, positions {layout.ending}")
        for m in DIMS:
            row = entry["ending"][str(m)][own - 1]
            print(f"{f'm = {m}':>12}  " + " ".join(f"{v:7.4f}" for v in row))

    # What every set on file must share to be read side by side.
    shared: dict[str, Any] = {
        "checkpoint": CHECKPOINT_NAME,
        "dims": list(DIMS),
        "primary_dim": PRIMARY_DIM,
        "jacobian_prompts": counted,
        "prefixes": [n for n in PREFIXES if n <= counted],
        "layout": {"head": layout.head, "tail": layout.tail},
        "site_layer": layer,
    }
    update: dict[str, Any] = {}

    if args.set == "reported":
        baselines = list(runs["baseline"].values())
        choice = choose_positions(baselines, layer)
        choice["largest_by_dim"] = {
            str(m): choose_positions(baselines, layer, m)["largest"] for m in DIMS
        }
        update["choice"] = choice
        print(f"\nbaseline at layer {layer}, m = {PRIMARY_DIM}, positions {choice['offsets']}:")
        print("  " + " ".join(f"{v:7.4f}" for v in choice["baseline"]))
        print(f"  largest at {choice['largest']} → align at {choice['positions']}"
              f"   (largest by width: {choice['largest_by_dim']})")

        gap = reproduces(runs, layer, counted)
        update["reproduces_probe_dependence"] = gap
        print("\naligned-site cell against probe_dependence.py: "
              + ("not compared" if gap is None else f"largest gap {gap:.2e}"))
        for arm in ARMS:
            means = [fmean(site_sensitivity(r, layer, (-1,), m) for r in runs[arm].values())
                     for m in DIMS]
            print(f"{arm:>10}" + "".join(f"{v:>10.4f}" for v in means))

    if counted != PROMPTS:
        print(f"\ntrial over {counted} prompts — nothing written.")
        raise SystemExit(0)

    result: dict[str, Any] = {**shared, "sets": {}}
    if OUT_PATH.exists():
        result = load_json(OUT_PATH)
        if differing := [key for key, value in shared.items() if result.get(key) != value]:
            raise SystemExit(
                f"{OUT_PATH.name} was measured under a different {', '.join(differing)}. "
                f"Sets measured differently cannot share a file; remove it and measure "
                f"every set again."
            )
    result["sets"][args.set] = runs
    result.update(update)
    save_json(result, OUT_PATH)
    print(f"\nwritten: {OUT_PATH}")
