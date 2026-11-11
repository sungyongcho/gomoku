"""Average the weights of several checkpoints into one model.

    python gomoku/scripts/average_checkpoints.py OUT CHECKPOINT [CHECKPOINT ...]

Paths may be local or fsspec URLs. See ``average_state_dicts`` for what is averaged.
"""

from __future__ import annotations

import io
import sys

import fsspec
import torch

from gomoku.alphazero.eval.reference_match import load_state_dict
from gomoku.utils.state_dict_utils import average_state_dicts


def main() -> None:
    if len(sys.argv) < 4:
        raise SystemExit(__doc__)
    out, paths = sys.argv[1], sys.argv[2:]
    averaged = average_state_dicts([load_state_dict(path) for path in paths])
    buffer = io.BytesIO()
    torch.save(averaged, buffer)
    with fsspec.open(out, "wb") as handle:
        handle.write(buffer.getvalue())
    print(f"averaged {len(paths)} checkpoints into {out} ({len(averaged)} tensors)")


if __name__ == "__main__":
    main()
