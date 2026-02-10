
import sys
import os
import re
import json
import glob
import torch
import numpy as np
from pathlib import Path

# Add alphazero root to sys.path so this script runs from any cwd.
ALPHAZERO_ROOT = Path(__file__).resolve().parent
sys.path.append(str(ALPHAZERO_ROOT))

from gomoku.core.gomoku import Gomoku
from gomoku.model.policy_value_net import PolicyValueNet
from gomoku.pvmcts.pvmcts import PVMCTS
from gomoku.utils.config.loader import MctsConfig, ModelConfig, BoardConfig
from gomoku.model.model_helpers import calc_num_planes


# ── Auto-detect run directory ────────────────────────────────────────

DEFAULT_RUNS_DIR = ALPHAZERO_ROOT / "runs"


def find_run_dir(runs_dir: str | Path = DEFAULT_RUNS_DIR) -> str | None:
    """Find the run directory with the most completed iterations."""
    runs_path = Path(runs_dir)
    if not runs_path.exists() or not runs_path.is_dir():
        return None

    candidates = []
    for entry in runs_path.iterdir():
        manifest_file = entry / "manifest.json"
        if not manifest_file.exists():
            continue
        try:
            with open(manifest_file, encoding="utf-8") as f:
                data = json.load(f)
            completed = int(data.get("progress", {}).get("completed_iterations", 0))
            candidates.append((completed, str(entry)))
        except Exception:
            continue
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


def load_manifest(run_dir: str) -> dict | None:
    manifest = os.path.join(run_dir, "manifest.json")
    if not os.path.exists(manifest):
        return None
    with open(manifest, encoding="utf-8") as f:
        return json.load(f)


def find_ckpt_dir(run_dir: str) -> str:
    """Return the checkpoint directory for a run."""
    ckpt = os.path.join(run_dir, "ckpt")
    if os.path.isdir(ckpt):
        return ckpt
    return run_dir


def discover_checkpoints(run_dir: str) -> list[tuple[str, str]]:
    """Auto-discover checkpoints from manifest + filesystem.

    Returns list of (label, path) sorted by iteration number.
    Always includes:
      - A baseline (first available among: pre-recovery, earliest)
      - The manifest champion
      - The latest checkpoint on disk (if different from champion)
    """
    manifest = load_manifest(run_dir)
    ckpt_dir = find_ckpt_dir(run_dir)

    # Scan all checkpoint files
    pattern = os.path.join(ckpt_dir, "iteration_*.pt")
    files = glob.glob(pattern)
    iter_map: dict[int, str] = {}
    for f in files:
        m = re.search(r"iteration_(\d+)\.pt$", f)
        if m:
            iter_map[int(m.group(1))] = f

    if not iter_map:
        return []

    all_iters = sorted(iter_map.keys())
    selected: dict[int, str] = {}  # iter -> label

    # 1. Baseline: iter 135 (pre-recovery) or the earliest available
    if 135 in iter_map:
        selected[135] = "iter 135 (pre-recovery baseline)"
    else:
        earliest = all_iters[0]
        selected[earliest] = f"iter {earliest} (earliest)"

    # 2. Champion from manifest
    champion_iter = None
    if manifest:
        champ_path = manifest.get("champion_model_path", "")
        m = re.search(r"iteration_(\d+)\.pt", champ_path)
        if m:
            champion_iter = int(m.group(1))

        # Also check promotions list for the last promoted iteration
        promotions = manifest.get("promotions", [])
        if promotions:
            last_promo = promotions[-1].get("iteration")
            if last_promo and last_promo in iter_map:
                champion_iter = last_promo

    if champion_iter and champion_iter in iter_map:
        elo_str = ""
        if manifest and "elo" in manifest:
            elo = manifest["elo"].get("champion", "?")
            elo_str = f", Elo {elo}"
        selected[champion_iter] = f"iter {champion_iter} (champion{elo_str})"

    # 3. Latest checkpoint on disk
    latest_iter = all_iters[-1]
    if latest_iter not in selected:
        selected[latest_iter] = f"iter {latest_iter} (latest on disk)"

    # Build sorted result
    result = []
    for it in sorted(selected.keys()):
        result.append((selected[it], iter_map[it]))
    return result


# ── Core logic (unchanged) ───────────────────────────────────────────

def load_checkpoint(model, ckpt_path, device):
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        try:
            model.load_state_dict(checkpoint)
        except Exception:
            new_state_dict = {}
            for k, v in checkpoint.items():
                if k.startswith("module."):
                    new_state_dict[k[7:]] = v
                else:
                    new_state_dict[k] = v
            model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()


def distance_from_center(x, y, center):
    return abs(x - center) + abs(y - center)


def run_opening_test(model, board_cfg, mcts_cfg, device, use_native_mcts, use_native_core):
    """Run MCTS on empty board, return list of (action, x, y, visits, prior)."""
    from gomoku.inference.local import LocalInference
    inference = LocalInference(model)

    game = Gomoku(board_cfg, use_native=use_native_core)
    state = game.get_initial_state()
    mcts = PVMCTS(game, mcts_cfg, inference, mode="sequential")
    root = mcts.create_root(state)
    mcts.engine.search([root], add_noise=False)

    board_size = board_cfg.num_lines
    results = []
    if root.children:
        for action, child in root.children.items():
            if isinstance(action, tuple) and len(action) == 2:
                x, y = action
            else:
                action_idx = action
                x, y = action_idx % board_size, action_idx // board_size
            results.append((action, x, y, child.visit_count, child.prior))
        results.sort(key=lambda r: r[3], reverse=True)
    return results, root.visit_count


def print_board_heatmap(results, board_size, top_n=10):
    """Print a simple text heatmap of top visited positions."""
    center = board_size // 2
    # Only show the central 11x11 region
    radius = 5
    lo = center - radius
    hi = center + radius

    visit_map = {}
    for _, x, y, visits, _ in results:
        if lo <= x <= hi and lo <= y <= hi:
            visit_map[(x, y)] = visits

    max_v = max(visit_map.values()) if visit_map else 1

    header = "     " + "".join(f"{c:>5}" for c in range(lo, hi + 1))
    print(header)
    for row in range(lo, hi + 1):
        cells = []
        for col in range(lo, hi + 1):
            v = visit_map.get((col, row), 0)
            if v == 0:
                cells.append("   . ")
            else:
                pct = v * 100 // max_v
                cells.append(f"{pct:>4}%")
        marker = " <" if row == center else ""
        print(f"  {row:>2} " + "".join(cells) + marker)
    print(f"       {'  ^':>{(center - lo) * 5 + 3}}")


def compare_checkpoints():
    board_size = 19
    history_length = 5
    num_planes = calc_num_planes(history_length)
    center = board_size // 2
    center_idx = center * board_size + center

    model_cfg = ModelConfig(
        num_hidden=128,
        num_resblocks=12,
        num_planes=num_planes
    )
    board_cfg = BoardConfig(
        num_lines=board_size,
        enable_doublethree=True,
        enable_capture=True,
        capture_goal=5,
        gomoku_goal=5,
        history_length=5
    )

    mcts_cfg = MctsConfig(
        C=2.0,
        num_searches=800,
        exploration_turns=0,
        dirichlet_epsilon=0.0,
        dirichlet_alpha=0.03,
        use_native=False,
        batch_infer_size=1,
        min_batch_size=1,
        max_batch_wait_ms=0
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Board: {board_size}x{board_size}, Center: ({center},{center}) idx={center_idx}")
    print(f"MCTS: {int(mcts_cfg.num_searches)} searches, no noise, deterministic\n")

    # Auto-detect run directory and checkpoints
    run_dir = find_run_dir()
    if not run_dir:
        print(f"No run directory found under: {DEFAULT_RUNS_DIR}")
        return

    manifest = load_manifest(run_dir)
    completed = "?"
    elo = "?"
    if manifest:
        completed = manifest.get("progress", {}).get("completed_iterations", "?")
        elo_raw = manifest.get("elo", {}).get("champion", "?")
        # elo can be a float or a dict like {"iteration": N, "elo": 1500.0}
        if isinstance(elo_raw, dict):
            elo = elo_raw.get("elo", "?")
        else:
            elo = elo_raw
    print(f"Run: {run_dir}  (completed={completed}, champion Elo={elo})")

    checkpoints = discover_checkpoints(run_dir)
    if not checkpoints:
        print("No checkpoints found!")
        return

    print(f"Checkpoints: {len(checkpoints)}")
    for label, path in checkpoints:
        print(f"  - {label}: {path}")
    print()

    # Collect results per checkpoint
    all_results = {}

    game_dummy = Gomoku(board_cfg, use_native=False)
    model = PolicyValueNet(game_dummy, model_cfg, device=device)

    for label, ckpt_path in checkpoints:
        print(f"{'=' * 70}")
        print(f"  {label}")
        print(f"  {ckpt_path}")
        print(f"{'=' * 70}")

        load_checkpoint(model, ckpt_path, device)

        results, total_visits = run_opening_test(
            model, board_cfg, mcts_cfg, device,
            use_native_mcts=False, use_native_core=False
        )
        all_results[label] = (results, total_visits)

        if not results:
            print("  No visits!\n")
            continue

        # Top 5 moves
        print(f"\n  Top 5 moves (out of {total_visits} total visits):")
        for i, (action, x, y, visits, prior) in enumerate(results[:5]):
            prob = visits / max(1, total_visits)
            dist = distance_from_center(x, y, center)
            is_center = (x == center and y == center)
            marker = " ** CENTER **" if is_center else ""
            print(f"    {i+1}. ({x:>2},{y:>2})  visits={visits:>5} ({prob:>6.1%})  prior={prior:.4f}  dist={dist}{marker}")

        # Center stats
        center_visits = 0
        center_prior = 0.0
        near_center_visits = 0  # dist <= 2
        for _, x, y, visits, prior in results:
            dist = distance_from_center(x, y, center)
            if dist == 0:
                center_visits = visits
                center_prior = prior
            if dist <= 2:
                near_center_visits += visits

        center_pct = center_visits / max(1, total_visits) * 100
        near_pct = near_center_visits / max(1, total_visits) * 100

        print(f"\n  Center ({center},{center}): {center_visits} visits ({center_pct:.1f}%), prior={center_prior:.4f}")
        print(f"  Near-center (dist<=2): {near_center_visits} visits ({near_pct:.1f}%)")

        # Top move distance from center
        top_x, top_y = results[0][1], results[0][2]
        top_dist = distance_from_center(top_x, top_y, center)
        print(f"  Top move distance from center: {top_dist}")

        # Heatmap
        print(f"\n  Visit heatmap (central 11x11, % of max):")
        print_board_heatmap(results, board_size)
        print()

    # Summary comparison
    if len(all_results) >= 2:
        print(f"\n{'=' * 70}")
        print("  COMPARISON SUMMARY")
        print(f"{'=' * 70}")
        print(f"  {'Checkpoint':<40} {'Center%':>8} {'Near%':>8} {'Top Move':>10} {'TopDist':>8}")
        print(f"  {'-'*40} {'-'*8} {'-'*8} {'-'*10} {'-'*8}")

        for label, (results, total_visits) in all_results.items():
            if not results:
                continue
            center_v = 0
            near_v = 0
            for _, x, y, visits, _ in results:
                dist = distance_from_center(x, y, center)
                if dist == 0:
                    center_v = visits
                if dist <= 2:
                    near_v += visits
            center_pct = center_v / max(1, total_visits) * 100
            near_pct = near_v / max(1, total_visits) * 100
            top_x, top_y = results[0][1], results[0][2]
            top_dist = distance_from_center(top_x, top_y, center)
            short_label = label[:40]
            print(f"  {short_label:<40} {center_pct:>7.1f}% {near_pct:>7.1f}% ({top_x:>2},{top_y:>2})  {top_dist:>6}")

        # Verdict
        labels = list(all_results.keys())
        first_results = all_results[labels[0]]
        last_results = all_results[labels[-1]]

        def get_center_pct(entry):
            results, total = entry
            for _, x, y, v, _ in results:
                if x == center and y == center:
                    return v / max(1, total) * 100
            return 0.0

        first_pct = get_center_pct(first_results)
        last_pct = get_center_pct(last_results)

        print()
        if last_pct > first_pct + 5:
            print(f"  VERDICT: Center opening IMPROVED ({first_pct:.1f}% -> {last_pct:.1f}%)")
        elif last_pct < first_pct - 5:
            print(f"  VERDICT: Center opening REGRESSED ({first_pct:.1f}% -> {last_pct:.1f}%)")
        else:
            print(f"  VERDICT: Center opening roughly UNCHANGED ({first_pct:.1f}% -> {last_pct:.1f}%)")
            if last_pct < 30:
                print(f"  WARNING: Center visit share is low (<30%). Model may still prefer off-center openings.")


if __name__ == "__main__":
    compare_checkpoints()
