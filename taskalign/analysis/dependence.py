"""Where the answer depends: §A.4's relative sensitivity, at every layer and position.

context.md §A.4 measured how hard the readout leans on one site — layer 16, the last
prompt token — and found that alignment loosened it. §8.5 asks the same of everywhere
else, to choose where to align next. The quantity is unchanged,

    e = |J^T u| * |h| / |r|

with `h` now the output of any block at any prompt position. One backward pass reaches
them all (`jacobian.vjp_everywhere`), so the directions, the prompts and the number of
backward passes are §A.4's own, and the cell at layer 16's last prompt token is §A.4's
number again.

**Positions are lined up along the prompt, not along the batch.** Every prompt is a fixed
head (the turn marker and the image), a question, and a fixed tail (the end of the turn
and the cue to answer); only the question varies in length. So the head is counted from
the front, the tail from the back, and the question from its first token — which puts the
question word of every prompt in one column. A question column further out is averaged
over the prompts long enough to reach it, and says how many those were.

**The tail is kept apart from the map, with the question's last token.** Those are the
positions that have read the whole question (context.md §8.5, second step) — the only
candidates for moving the alignment — and they are kept at every width and at every
prefix of the prompt set, where the map itself is kept at one width only.

**A first-order sensitivity is not conserved.** Each cell is how far the readout moves
when that cell alone does. The cells do not sum to anything, so a cell that fell says
nothing about another that rose; the map chooses a site, and does not show dependence
moving from one place to another.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from statistics import fmean
from typing import Any

import torch

from ..models import SmolVLM
from .jacobian import vjp_everywhere

__all__ = [
    "BATCH",
    "DIMS",
    "PREFIXES",
    "PRIMARY_DIM",
    "PROMPTS",
    "Layout",
    "choose_positions",
    "measure_map",
    "site_sensitivity",
]

# §A.4's own: the widths of the answer subspace, the prompts and their batching, and the
# prefixes at which the running value shows whether the prompts were enough.
DIMS     = (8, 32, 128)
PROMPTS  = 64
BATCH    = 8
PREFIXES = (8, 16, 32, 48, 64)

# The width a choice or a verdict is read at, fixed before the map was measured
# (context.md §8.5, 2026-11-20). §A.4 reports three widths and a rule needs one: the
# middle one is neither the width at which alignment's effect was largest (8) nor the
# one at which it was smallest (128). The other two are reported beside it.
PRIMARY_DIM = 32


@dataclass(frozen=True)
class Layout:
    """How many prompt tokens come before the question, and how many after it."""

    head: int
    tail: int

    @property
    def ending(self) -> tuple[int, ...]:
        """The positions that have read the whole question, counted back from the
        prompt's end: the question's last token, then the tail. -1 is the last prompt
        token.
        """
        return tuple(range(-self.tail - 1, 0))


def measure_map(
    vlm: SmolVLM,
    batches: list[dict[str, Any]],
    directions: torch.Tensor,
    layout: Layout,
    layer: int,
) -> dict[str, Any]:
    """One run's relative sensitivity, averaged over directions and prompts.

    `ending` is `(L, E)` per width — block by position, over `layout.ending` — and
    `running` its row at `layer`, the layer the run aligned, at each prefix of the prompt
    set. `map` is the rest of the prompt at `PRIMARY_DIM`: `head`, `(L, head)`, and
    `question`, `(L, Q)` over the longest question, with `question_prompts` saying how
    many prompts each of its columns averages. Block `l` is row `l - 1`.
    """
    primary = DIMS.index(PRIMARY_DIM)
    ending: list[torch.Tensor] = []
    heads: list[torch.Tensor] = []
    questions: list[torch.Tensor] = []
    for batch in batches:
        lengths, hidden_norm, readout_norm = vjp_everywhere(vlm, batch, directions)
        sensitivity = lengths * hidden_norm.unsqueeze(1) / readout_norm.view(-1, 1, 1, 1)
        for row, end in enumerate(batch["prompt_lens"]):
            # (D, L, T): the directions averaged away, one slice per width.
            by_dim = torch.stack([sensitivity[row, :m].mean(dim=0) for m in DIMS])
            ending.append(by_dim[:, :, end + layout.ending[0] : end])
            heads.append(by_dim[primary, :, : layout.head])
            questions.append(by_dim[primary, :, layout.head : end - layout.tail])

    per_prompt = torch.stack(ending)  # (P, D, L, E)
    longest = max(question.size(1) for question in questions)
    columns = [
        torch.stack([q[:, k] for q in questions if q.size(1) > k]) for k in range(longest)
    ]
    return {
        "layer": layer,
        "ending": {
            str(m): _rounded(per_prompt[:, d].mean(dim=0)) for d, m in enumerate(DIMS)
        },
        "running": {
            str(m): [
                _rounded(per_prompt[:n, d, layer - 1].mean(dim=0))
                for n in PREFIXES if n <= len(per_prompt)
            ]
            for d, m in enumerate(DIMS)
        },
        "map": {
            "head": _rounded(torch.stack(heads).mean(dim=0)),
            "question": _rounded(torch.stack([c.mean(dim=0) for c in columns], dim=1)),
            "question_prompts": [len(column) for column in columns],
        },
    }


def site_sensitivity(
    run: dict[str, Any], layer: int, positions: Sequence[int], dim: int = PRIMARY_DIM
) -> float:
    """A run's relative sensitivity at a site: one block, averaged over `positions`.

    `positions` count back from the prompt's end, as the model's own do, so the reported
    site is `(16, (-1,))` and reads §A.4's number.
    """
    row = run["ending"][str(dim)][layer - 1]
    return fmean(row[position] for position in positions)


def choose_positions(
    baselines: Sequence[dict[str, Any]], layer: int, dim: int = PRIMARY_DIM
) -> dict[str, Any]:
    """Where to align next, by the rule context.md §8.5 fixed before the map existed.

    Among the positions that have read the whole question, the one the baseline's answer
    depends on most, at the layer already aligned. If that is the last prompt token —
    the position already aligned — moving there moves nothing, so all of them are
    aligned instead. The baseline is averaged over its seeds first: the choice is one
    site for every seed, not a site apiece.
    """
    rows = [run["ending"][str(dim)][layer - 1] for run in baselines]
    mean = [fmean(column) for column in zip(*rows, strict=True)]
    offsets = list(range(-len(mean), 0))
    largest = offsets[max(range(len(mean)), key=mean.__getitem__)]
    return {
        "layer": layer,
        "dim": dim,
        "offsets": offsets,
        "baseline": mean,
        "largest": largest,
        "positions": offsets if largest == -1 else [largest],
    }


def _rounded(values: torch.Tensor) -> Any:
    """Nested lists at six decimals — the map is read as a picture, and a relative
    sensitivity is of order 0.1, so nothing the document quotes lives below that.
    """
    return (values * 1e6).round().div(1e6).tolist()
