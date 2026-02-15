# Documentation Plan: Minimax & AlphaZero Technical Docs

## Context

This project is a portfolio piece demonstrating full-stack Gomoku AI development. The documentation targets recruiters, CTOs, and engineers evaluating the developer's skills. The live demo is playable at **sungyongcho.com/gomoku** — this is the strongest card and should be prominent.

**Structure**: Topic-based with embedded narrative. Organized by technical topic, but the journey (Python→C++, original repo → refactored repo, GCP training) is woven in naturally as "why" context. Every section implicitly demonstrates a skill.

**Language**: English. Technical but accessible — practitioner explaining decisions, not a textbook. "I chose X because Y" over "X is defined as..."

**Math**: KaTeX inline (`$...$`) and block (`$$...$$`).

**Diagrams**: Leave `<!-- DIAGRAM: description -->` placeholders with detailed descriptions. The developer will create diagrams separately.

**Code snippets**: Maximum 3 per file. Pick the ones that make someone think "that's clever" — not the obvious ones.

**Tone formula for every section**: Lead with the problem → explain the solution → show a focused snippet or formula → note what was learned or what tradeoff was made.

---

## References (hyperlinks to include in appropriate docs)

- Alpha-beta pruning (Korean): https://going-to-end.tistory.com/entry/%EC%95%8C%ED%8C%8C-%EB%B2%A0%ED%83%80-%EA%B0%80%EC%A7%80%EC%B9%98%EA%B8%B0-Alpha-beta-pruning
- Minimax / Alpha-beta overview (YouTube): https://www.youtube.com/watch?v=AM-4drnj63w
- Principal Variation Search: https://www.chessprogramming.org/Principal_Variation_Search
- CForbiddenPointFinder source (renju.se): https://www.renju.se/renlib/opensrc/
- AlphaZero tutorial (YouTube, foersterrobert): https://www.youtube.com/watch?v=wuSQpLinRB4
- AlphaZeroFromScratch repo: https://github.com/foersterrobert/AlphaZeroFromScratch

---

## File Structure

### Minimax (`front/content/docs/minimax/`) — 4 files

- [x] `00.intro.md` — Introduction: Motivation and Board Representation (`pi-bolt`)
- [ ] `01.search-algorithms.md` — Search: From Alpha-Beta to PVS (`pi-search`)
- [ ] `02.evaluation.md` — Evaluation: Pattern Lookup Tables (`pi-chart-bar`)
- [ ] `03.rules-and-serving.md` — Game Rules, Captures, and Serving (`pi-shield`)

**Why 4 files, not 6**: Minimax is a solid implementation but it's standard game AI (alpha-beta, TT, killer moves). The differentiation in this project is the AlphaZero system, Ray pipeline, and deployment. Minimax should be tight, sharp, and respect the reader's time. Board representation merges into intro (bitboard IS the motivation story). Rules and serving merge (both are concise, and rules feed directly into the serving layer).

### AlphaZero (`front/content/docs/alphazero/`) — 8 files

- [ ] `00.intro.md` — Introduction and Background (`pi-bolt`)
- [ ] `01.neural-network.md` — The Policy-Value Network (`pi-microchip`)
- [ ] `02.mcts.md` — Monte Carlo Tree Search with PUCT (`pi-sitemap`)
- [ ] `03.search-engine-architecture.md` — Search Engine Architecture: From Sequential to Distributed (`pi-cog`)
- [ ] `04.training-pipeline.md` — Training Pipeline and GCP Infrastructure (`pi-sync`)
- [ ] `05.evaluation.md` — Model Evaluation: Arena and SPRT (`pi-chart-line`)
- [ ] `06.cpp-extensions.md` — C++ Extensions: pybind11 for Performance (`pi-bolt`)
- [ ] `07.deployment.md` — Production Deployment and Lessons Learned (`pi-cloud`)

**Why 8, not 9**: Search engines (was 03) and Ray async pipeline (was 04) merge into a single "Search Engine Architecture" page. The overview of 4 modes is too thin to stand alone, and the Ray async deep dive needs the context of the other modes to make sense. One page: brief overview of all modes → deep dive into the async pipeline as the most architecturally interesting one.

### Files to Delete (existing stubs)
- [x] `front/content/docs/minimax/01.content.md`
- [x] `front/content/docs/alphazero/01.content.md`

---

## Content Outline per File

### Minimax

---

#### `00.intro.md` — Introduction: Motivation and Board Representation

This file does two things: (1) explain what minimax is and why C++ was chosen, and (2) introduce the bitboard representation that makes everything else possible. These belong together because the bitboard IS the motivation — it's why C++ was worth the effort.

**Section 1: What is Minimax Search** (2-3 paragraphs)
- Brief explanation: two-player zero-sum game tree, alternating max/min layers
- The minimax value formula in KaTeX:

$$v(s) = \max_{a \in A(s)} \min_{b \in B(s)} v(s')$$

- Why minimax fits Gomoku: perfect information, zero-sum, deterministic
- But: 19x19 board means branching factor ~200+, so naive minimax is impossible at meaningful depth

**Section 2: Why C++** (2-3 paragraphs)
- **Narrative**: The Python implementation was functional but too slow, especially for double-three detection on a 19x19 board. C++98 was chosen because the developer was comfortable with it from École 42 projects — a personal choice combining familiarity with performance.
- The key insight: represent the board as bitboards (`uint64_t[19]` per player), enabling bitwise operations for move generation, pattern matching, and evaluation instead of array iteration.
- Even after the bitmask-based double-three detection was later proven incorrect and removed, the bitboard representation was kept because it still provides massive performance for the evaluation pattern matching (which is the real hot path).

**Section 3: Bitboard Design** (2-3 paragraphs)
- Two arrays of `uint64_t[19]`, one per player. Each uint64 = one row, bits 0-18 = columns.
- Why separate bitboards (not tri-state): fast OR for occupancy, fast AND/SHIFT for pattern extraction in the evaluation function.
- Move generation: compute neighbor masks via bit shifts (horizontal, vertical, diagonal), AND with NOT-occupancy to find candidate moves adjacent to existing stones. This reduces candidates from 361 to ~20-40 per position.
- Zobrist hashing for transposition table: `piece_keys[19][19][3]` with XOR-incremental updates (never recomputed from scratch). Boost MT19937 RNG for key generation.
- Undo/Redo via `UndoInfo` struct: in-place board mutation during search, no board copy at each node.

**Section 4: Difficulty Modes** (table)

| Difficulty | Search Algorithm | Max Depth | Time Limit | Eval Function |
|------------|-----------------|-----------|------------|---------------|
| Easy | Alpha-Beta | 5 | None | Simple |
| Medium | Iterative Deepening | 10 | 0.4s | Simple |
| Hard | PVS | 10 | None | Hard (pattern-counted) |

**Section 5: References**
- Link to [alpha-beta pruning blog (Korean)](https://going-to-end.tistory.com/entry/%EC%95%8C%ED%8C%8C-%EB%B2%A0%ED%83%80-%EA%B0%80%EC%A7%80%EC%B9%98%EA%B8%B0-Alpha-beta-pruning) and [minimax/alpha-beta YouTube video](https://www.youtube.com/watch?v=AM-4drnj63w)

<!-- DIAGRAM: Architecture overview: WebSocket Request → JSON Parser → Board (bitboard) → Search Engine → Evaluation (Pattern Lookup Table) → Response. Show the data flow linearly. -->

<!-- DIAGRAM: Row-to-uint64 bit mapping. Show a 19-cell row and how each cell maps to a bit position in a uint64_t. Mark bits 0-18 used, bits 19-63 unused. -->

**Code blocks (pick best 3):**
- `minimax/inc/gomoku/Board.hpp:53-54` — bitboard declaration (the core data structure, 2 lines that define the whole approach)
- `minimax/src/gomoku/search/Minimax.cpp:25-43` — `computeNeighborMask` (the bitwise neighbor computation — this is the clever part that shows why bitboards matter)
- `minimax/src/ws/request_handlers.cpp:14-29` — `selectBestMove` difficulty branching (shows the three modes cleanly)

---

#### `01.search-algorithms.md` — Search: From Alpha-Beta to PVS

This is the technical heart of the minimax section. The narrative arc: alpha-beta alone wasn't enough → added enhancements one by one → each one unlocked more depth.

**Section 1: Alpha-Beta Pruning** (2-3 paragraphs)
- The core idea: maintain an alpha-beta window. If a node's score can't influence the final decision, prune.
- KaTeX: $\alpha$-$\beta$ cutoff: if $\alpha \geq \beta$, prune remaining children
- Complexity: reduces from $O(b^d)$ to $O(b^{d/2})$ with perfect move ordering
- **Narrative**: This was the first search algorithm implemented. It improved over naive minimax dramatically but wasn't enough for depth 10 on a 19x19 board in reasonable time. The key bottleneck: move ordering.

**Section 2: Move Ordering** (2-3 paragraphs)
- Alpha-beta effectiveness depends entirely on examining the best move first. Three tiers:
  1. **TT hash move**: If a previous search found a best move for this position, try it first.
  2. **Killer moves**: Two per depth level — moves that caused beta cutoffs at sibling nodes. Stored in `killerMoves[MAX_DEPTH+1][2]`, updated on every cutoff.
  3. **Evaluation-sorted**: Remaining moves scored by the evaluation function and sorted descending.
- This ordering is what makes depth 10 feasible: the TT move is often the PV move from a shallower iteration, causing immediate cutoffs.

**Section 3: Transposition Table** (2 paragraphs)
- `boost::unordered_map<uint64_t, TTEntry>` mapping Zobrist hash to `{score, depth, bestMove, boundType}`.
- Three bound types: EXACT (score within window), LOWERBOUND (score ≥ beta, caused cutoff), UPPERBOUND (score ≤ alpha, all moves failed). On lookup, the stored bound is checked against the current window.
- The TT also serves as the hash move source for move ordering — its best move from a previous search is tried first.

**Section 4: Principal Variation Search (PVS)** (3 paragraphs)
- The insight: assume the first child (the "principal variation") is the best move. Search it with a full window. Search all subsequent children with a **null window** ($-\alpha-1, -\alpha$) — essentially asking "is this move better than my current best?"
- If a null-window search returns a score $> \alpha$ and $< \beta$: the PV assumption was wrong. Re-search with the full window.
- **Narrative**: PVS was implemented with help from the [chessprogramming.org PVS article](https://www.chessprogramming.org/Principal_Variation_Search) and LLM assistance for the null-window re-search logic. Understanding the theory was one thing; getting the sign conventions and bound types right in practice was the real challenge.

**Section 5: Iterative Deepening** (1-2 paragraphs)
- Search depth 1, then 2, then 3... up to maxDepth or time limit. Not wasteful: each iteration populates the TT, so deeper iterations benefit from better move ordering.
- Medium difficulty uses this with a 0.4s time limit. On timeout, returns the best move from the deepest completed iteration.

**Section 6: Quiescence Search** (1-2 paragraphs)
- At depth 0, instead of returning a static evaluation, continue searching capture-only moves. Prevents the "horizon effect" where a devastating capture is just beyond the search depth.
- Stand-pat score serves as a lower bound for the maximizing player.

<!-- DIAGRAM: Game tree showing alpha-beta pruning. 3-4 levels, with pruned branches marked. Show alpha/beta values at each node. -->

<!-- DIAGRAM: PVS null-window illustration. Show first child searched with full window [α, β], subsequent children with null window [α, α+1], and one child that triggers re-search. -->

**Code blocks (pick best 3):**
- `minimax/src/gomoku/search/Minimax.cpp:578-650` — `pvs` function (the null-window search + re-search logic — this is the most technically interesting algorithm)
- `minimax/src/gomoku/search/Minimax.cpp:258-283` — `probeTT` (shows the three bound types and how they interact with the current window)
- `minimax/src/gomoku/search/Minimax.cpp:357-360` — killer move update (just 4 lines but shows the elegant simplicity of the heuristic)

---

#### `02.evaluation.md` — Evaluation: Pattern Lookup Tables

This section showcases the most original engineering in the minimax: turning the evaluation problem into a pre-computed O(1) lookup.

**Section 1: The Evaluation Problem** (1-2 paragraphs)
- At every leaf node, the search needs a static score. On a 19x19 board with depth 10, the evaluation function is called millions of times per move. It must be fast.
- What needs scoring: continuous stone lines (open threes, open fours, gomoku), blocking threats, capture opportunities, capture vulnerability, positional advantage.

**Section 2: Window-Based Pattern Extraction** (2-3 paragraphs)
- For each position, extract a 9-cell window along each of 4 axes (horizontal, vertical, two diagonals): 4 cells backward + center + 4 cells forward.
- Each cell is encoded in 2 bits: `00`=empty, `01`=player 1, `10`=player 2, `11`=out-of-bounds. The 9-cell window (minus the fixed center) becomes an 18-bit integer.
- This encoding uses the bitboard directly — pattern extraction is a series of bit operations, not array lookups.

**Section 3: Pre-Computed Lookup Table** (2-3 paragraphs)
- At server startup, all $2^{18} = 262{,}144$ possible 18-bit patterns are evaluated once and stored in score arrays (`patternScoreTablePlayerOne[262144]`, `patternScoreTablePlayerTwo[262144]`).
- At search time, evaluation = extract 18-bit pattern per axis, index the table, sum. Four lookups total:

$$\text{score}(x,y) = \sum_{d \in \{H, V, D1, D2\}} \text{table}[\text{pattern}(x, y, d)]$$

- This is the payoff of the bitboard representation: the evaluation that dominates runtime is reduced to O(1) per axis.

**Section 4: Simple vs Hard Evaluation** (2-3 paragraphs)
- **Simple eval** (easy/medium): Uses raw lookup table scores. Fast, sufficient for lower difficulties.
- **Hard eval** (hard difficulty): Beyond raw scores, counts specific pattern types using `PatternCounts` struct — open fours, closed fours, open threes, etc. Then applies weighted scoring with nuanced logic:
  - Capture vulnerability penalty (penalizes stones that can be sandwiched)
  - Perfect critical line detection (gomoku on lines that can't be captured)
  - Capture-on-critical-line bonuses (breaking opponent's 4-in-a-row by capturing)
  - Center bonus with distance falloff
- Score hierarchy: CAPTURE_WIN (22M) > GOMOKU (21M) > BLOCK_GOMOKU (2M) > ... > CENTER_BONUS (1K)
- These values were hand-tuned through playtesting. Getting the relative priorities right (e.g., should blocking a 4-in-a-row score higher than making your own open 3?) is what makes the hard evaluation strong.

**Section 5: What Worked and What Didn't** (1-2 paragraphs)
- **Narrative**: The bitboard representation was originally also used for bitmask-based double-three detection. This approach was elegant but was proven incorrect in certain edge cases during the AlphaZero implementation — the AlphaZero's game engine exposed positions where the bitmask pattern matching produced wrong results. The double-three bitmask logic was removed and replaced with CForbiddenPointFinder (see next section). However, the evaluation bitmask pattern matching remains and works correctly — it's the performance foundation for the entire search.

<!-- DIAGRAM: 9-cell window extraction along one axis. Show a board position, the 4+1+4 cells selected along a diagonal, and the resulting 18-bit encoding with 2-bit-per-cell mapping. -->

**Code blocks (pick best 3):**
- `minimax/src/gomoku/eval/Evaluation.cpp:324-369` — `evaluateCombinedAxis` (the pattern extraction + lookup — this is the core insight)
- `minimax/src/gomoku/eval/Evaluation.cpp:280-304` — `initCombinedPatternScoreTables` (the startup pre-computation)
- `minimax/inc/gomoku/Evaluation.hpp:9-24` — score constants (shows the hand-tuned priority hierarchy)

---

#### `03.rules-and-serving.md` — Game Rules, Captures, and Serving

Compact section covering rule implementation and the WebSocket server. These belong together: the rules are what make this Gomoku variant unique, and the server is how they're exposed.

**Section 1: Capture Detection** (1-2 paragraphs)
- Sandwich pattern: placing stone P where `P-OO-P` exists in any of 8 directions captures the two middle opponent stones. Bitmask-based: check `(O[y1] & mask1) && (O[y2] & mask2) && (P[y3] & mask3)` per direction.

**Section 2: Double-Three (Forbidden Move) Detection** (2-3 paragraphs)
- A move is forbidden if it creates 2+ "open threes" simultaneously.
- `CForbiddenPointFinder` — sourced from [renju.se open source](https://www.renju.se/renlib/opensrc/). Originally, the team implemented bitmask pattern-matching functions for double-three detection, but this was found to produce incorrect results in certain edge cases during the AlphaZero implementation. After research, the CForbiddenPointFinder from renju.se was adopted as a well-tested implementation.
- **Important optimization for search**: double-three is checked per candidate move only, not a full board scan. `shouldIncludeMove` calls `Rules::detectDoublethree()` → `finder.IsDoubleThree(x, y)` — checking only the specific position. The full-board `FindForbiddenPoints()` exists but is only used for game state display, not during search.
- Exception: if a double-three move also captures stones, it's allowed (capture resolves the forbidden state).

**Section 3: Win Condition** (1 paragraph)
- 5+ stones in a row OR capturing 5 pairs (10 stones). If a gomoku line contains stones vulnerable to capture, it's scored lower ("breakable gomoku") in the evaluation.

**Section 4: WebSocket Server** (1-2 paragraphs)
- libwebsockets server on port 8005. Stateless per request: the entire board state is sent in the JSON payload each time.
- On startup: initialize Zobrist keys + pre-compute both evaluation lookup tables. At request time: parse JSON → construct Board → select search by difficulty → run search → return AI move, updated board, captured stones, eval scores, execution time.
- Reference the WebSocket protocol specification from the About Gomoku docs — don't duplicate.

**Code blocks (pick best 3):**
- `minimax/src/gomoku/search/Minimax.cpp:77-83` — `shouldIncludeMove` (the double-three filter with capture exception — shows elegant rule integration into search)
- `minimax/src/gomoku/forbidden/ForbiddenPointFinder.cpp:1037-1052` — `IsDoubleThree` (the core forbidden point check)
- `minimax/src/gomoku/core/Rules.cpp:66-82` — `detectDoublethree` (shows how board is reconstructed for the finder)

---

### AlphaZero

---

#### `00.intro.md` — Introduction and Background

This page sets up the entire AlphaZero section. It must answer: "Why AlphaZero? What was built? Does it work?" The live demo link goes here.

**Section 1: From Minimax to AlphaZero** (2-3 paragraphs)
- The minimax engine plays well but is fundamentally limited: every heuristic is hand-crafted and hand-tuned. The evaluation function cannot adapt or improve on its own. Adding capture rules required manually retuning score constants.
- AlphaZero (Silver et al., 2018) demonstrated that a neural network + MCTS can learn to play from self-play alone, with zero domain-specific evaluation knowledge beyond the rules.
- The live result is playable at **sungyongcho.com/gomoku** — select AlphaZero in game settings.

**Section 2: Learning Path** (2-3 paragraphs)
- **Narrative**: Read the AlphaGo paper (Silver et al., 2016) and AlphaZero paper (Silver et al., 2018). Found [foersterrobert's AlphaZero tutorial (YouTube)](https://www.youtube.com/watch?v=wuSQpLinRB4) and [AlphaZeroFromScratch repo](https://github.com/foersterrobert/AlphaZeroFromScratch), which provided a practical implementation starting point.
- Implemented sequential and vectorize search modes based on this tutorial, then added multiprocessing and Ray distributed training.
- The original implementation (in a separate repository) worked for small boards but revealed data integrity issues at scale, which motivated a complete architectural refactoring that produced the current modular codebase.

**Section 3: Architecture Overview** (1 paragraph + diagram)
- System components: Game Engine (rules + state management) → Neural Network (policy + value predictions) → MCTS (tree search guided by the network) → Training Pipeline (self-play → replay buffer → gradient updates → evaluation → model promotion) → Serving Layer (FastAPI + WebSocket).

**Section 4: Key Numbers** (bullet list — these are the "so what" proof points)
- 19×19 board, 361 possible actions per turn
- 128-channel ResNet with 12 residual blocks
- 13-channel state encoding
- 200-1600 MCTS simulations per move during training, 20 in production (native C++)
- 215 training iterations on GCP Ray cluster
- ~4160 cumulative Elo
- Observable behavior: started blocking opponent patterns around training day 3, developed aggressive play patterns (creating open threes) in the current champion model

**No explicit bug list.** The refactoring narrative shows engineering maturity. Specific design decisions (e.g., "PER requires updating priorities after each training step") are mentioned in context within the training pipeline section, framed as design knowledge.

<!-- DIAGRAM: High-level system architecture. Show the AlphaZero training loop as a cycle: Self-Play (generate games) → Replay Buffer (store samples) → Training (backprop) → Evaluation (arena match) → Promotion (update champion) → back to Self-Play. Show Serving as a separate branch from the trained model. -->

**Code blocks:** None — this is an overview/narrative page.

---

#### `01.neural-network.md` — The Policy-Value Network

**Section 1: Dual-Head Architecture** (2-3 paragraphs)
- A single CNN processes the board state and produces two outputs:
  - **Policy head**: probability distribution over 361 actions (where to play)
  - **Value head**: scalar in [-1, 1] predicting who is winning
- This replaces both the evaluation function (value) and move ordering (policy) from classical search.
- Combined loss:

$$L = (z - v)^2 - \pi^T \log p + c\|\theta\|^2$$

  where $z$ = game outcome, $v$ = predicted value, $\pi$ = MCTS policy (visit counts), $p$ = network policy, $c\|\theta\|^2$ = L2 regularization.

**Section 2: ResNet Architecture** (2-3 paragraphs)
- Start block: `Conv2d(13→128, 3×3, pad=1)` → BatchNorm → ReLU
- 12 residual blocks: `Conv(128→128, 3×3)` → BN → ReLU → `Conv(128→128, 3×3)` → BN → skip connection → ReLU
- Policy head: `Conv(128→2, 1×1)` → BN → ReLU → Flatten(2×19×19=722) → `Linear(722→361)` → logits
- Value head: `Conv(128→1, 1×1)` → BN → ReLU → Flatten(361) → `Linear(361→256)` → ReLU → `Linear(256→1)` → Tanh
- Why 128 channels / 12 blocks: The original AlphaZero paper used 256 channels and 19-39 residual blocks with TPU-scale compute. 128/12 is the balance point for the available GPU budget while still providing sufficient model capacity for 19×19 Gomoku.

**Section 3: State Encoding** (2-3 paragraphs)
- 13 input channels (each 19×19):
  - Channel 0: current player's stones (binary)
  - Channel 1: opponent's stones (binary)
  - Channel 2: empty positions (binary)
  - Channel 3: last move indicator (one-hot)
  - Channels 4-5: capture scores (current player ratio, opponent ratio, clipped 0-1)
  - Channel 6: color indicator (1.0 for Black, -1.0 for White — tells the network who is playing)
  - Channel 7: forbidden move mask (double-three positions, binary)
  - Channels 8-12: move history (last 5 moves as one-hot planes)
- This encoding gives the network everything it needs: the current position, game-specific context (captures, forbidden moves), temporal context (move history), and player identity.

<!-- DIAGRAM: Network architecture. Show vertically: Input (13×19×19) → Start Block → 12× Residual Block → split into Policy Head (→ 361 logits) and Value Head (→ scalar). Label layer sizes at each stage. -->

**Code blocks (pick best 3):**
- `alphazero/gomoku/model/policy_value_net.py` — `__init__` method (shows the full architecture construction: start_block, backbone, policy_head, value_head)
- `alphazero/gomoku/core/gomoku.py` — `get_encoded_state` method (shows the 13-channel encoding — the "what the network sees")
- `alphazero/gomoku/model/policy_value_net.py` — `forward` method (shows the dual-head split)

---

#### `02.mcts.md` — Monte Carlo Tree Search with PUCT

**Section 1: MCTS vs Minimax** (1-2 paragraphs)
- Unlike minimax which exhaustively searches to a fixed depth, MCTS builds a search tree asymmetrically — spending more time on promising moves.
- After N simulations, the action policy is proportional to visit counts at the root's children. More visits = more confidence = better move.

**Section 2: Four Phases** (2-3 paragraphs)
- **Select**: Traverse from root to leaf using UCB to balance exploration vs exploitation.
- **Expand**: Add the leaf's children to the tree.
- **Evaluate**: Run neural network inference on the leaf state → get policy priors + value estimate.
- **Backup**: Propagate the value estimate from leaf back to root, flipping sign at each level (opponent's gain is our loss).

**Section 3: PUCT Selection Formula** (2-3 paragraphs)
- The key formula:

$$UCB(s,a) = Q(s,a) + c_{\text{puct}} \cdot P(s,a) \cdot \frac{\sqrt{N(s)}}{1 + N(s,a)}$$

  where $Q$ = mean value of child, $P$ = prior from neural network, $N$ = visit count.
- This balances exploitation (high Q) with exploration (high prior, low visits). The neural network's policy prior guides the search toward promising moves before they've been visited — this is what makes MCTS+NN so powerful compared to MCTS with random rollouts.
- Sign flip: `q_value = -child.q_value` because the child represents the opponent's perspective.

**Section 4: Dirichlet Noise** (1-2 paragraphs)
- At the root only, Dirichlet noise is added to priors:

$$P'(s,a) = (1 - \epsilon) \cdot P(s,a) + \epsilon \cdot \eta_a, \quad \eta \sim \text{Dir}(\alpha)$$

- This ensures exploration during self-play: even if the network strongly favors one move, noise gives other moves a chance. Parameters: $\epsilon = 0.25$, $\alpha = 0.15$.

**Section 5: Temperature-Based Action Selection** (1 paragraph)
- During the first 20 moves (exploration turns), actions are sampled proportionally:

$$\pi(a) \propto N(a)^{1/\tau}$$

  with $\tau = 1.0$. After exploration turns, $\tau \to 0$ (argmax — pick the most-visited move).

<!-- DIAGRAM: MCTS 4-phase loop. Show a tree with root and children. Label the four phases with arrows: (1) Select path down to leaf, (2) Expand leaf into children, (3) Evaluate leaf with NN → policy + value, (4) Backup value to root with alternating signs. -->

**Code blocks (pick best 3):**
- `alphazero/gomoku/pvmcts/treenode.py` — `get_ucb` method (the PUCT formula in code — maps directly to the math)
- `alphazero/gomoku/pvmcts/treenode.py` — `expand` method (shows legal mask filtering + child creation from policy)
- `alphazero/gomoku/pvmcts/treenode.py` — `backup` method (sign-flipping value propagation — the tricky part)

---

#### `03.search-engine-architecture.md` — Search Engine Architecture: From Sequential to Distributed

This page has two parts: (1) brief overview of the four search engine modes (strategy pattern), and (2) deep dive into the Ray async pipeline — the most architecturally complex part of the project.

**Part 1: Strategy Pattern Overview** (3-4 paragraphs)

The problem: MCTS requires many neural network inferences per move. The optimal strategy depends on the hardware: single CPU, local GPU, multiple processes, or a distributed Ray cluster. Rather than hardcoding one approach, the system uses a strategy pattern: all engines implement the `SearchEngine` ABC, and the `PVMCTS` facade selects the right one based on configuration.

Four modes:
- **Sequential**: One simulation at a time, synchronous NN inference. For development, debugging, and CPU serving. Supports native C++ MCTS (`CppSearchStrategy`).
- **Vectorize**: Interleaves multiple tree traversals and batches leaf evaluations into a single GPU call. Optimizes GPU utilization for local training. Does NOT support native C++ MCTS.
- **Multiprocess**: Distributes MCTS simulations across a process pool with a central inference server via IPC. Falls back to SequentialEngine when native C++ is enabled.
- **Ray**: Distributed across a Ray cluster. The most complex mode — uses an asynchronous pipelined architecture to overlap CPU tree traversal with remote GPU inference. Deep dive below.

Native C++ MCTS compatibility:
- Sequential: ✅ full support
- Ray: ✅ via SequentialEngine + CppSearchStrategy with async Ray callbacks (confirmed used in elo1800-v4 training configs)
- Vectorize: ❌ not supported
- Multiprocess: ⚠️ falls back to single-threaded SequentialEngine

**Part 2: Asynchronous Pipelined MCTS with Ray** (the deep dive)

**The core problem** (1-2 paragraphs)
- A synchronous (stop-and-wait) approach — send batch to remote GPU actor, block until result returns, then continue tree traversal — wastes both CPU and GPU. The GPU sits idle during tree traversal, and the CPU sits idle during inference. With network latency to remote GPU actors, this becomes the dominant bottleneck.

**The solution: pipelined architecture** (1 paragraph)
- Overlap CPU work (tree selection/expansion) with GPU work (inference). While the GPU processes batch N, the CPU selects leaf nodes for batch N+1. This requires solving the **duplicate path problem**: without protection, multiple selections target the same promising leaf before the first result returns.

**Three components** (2-3 paragraphs each):

1. **RayAsyncEngine** (`ray_async.py`): Orchestrates the main search loop. Three phases per iteration:
   - **Selection phase**: Traverse tree via UCB to find leaf nodes, apply virtual loss (`pending_visits += 1` on every node in the path), encode state tensor, enqueue to batch manager. Guards: skip if leaf is already in-flight (`inflight_to_root` dict), pause if queue pressure exceeds `max_pending` (backpressure).
   - **Dispatch phase**: Send ready batches to Ray actors (delegates to BatchInferenceManager).
   - **Drain phase**: Collect results via `ray.wait(num_returns=1)` — returns as soon as ANY batch completes, not waiting for all. For each result: expand leaf (create children from policy), backup value to root (sign-flipping), remove virtual loss (`pending_visits -= 1`). Freed inflight slots trigger `dispatch_ready()` to send more batches immediately.

2. **BatchInferenceManager** (`batch_inference_manager.py`): Manages the batching queue with adaptive dispatch:
   - Dispatch condition (OR logic): queue size ≥ `batch_size` OR elapsed time since first enqueued item ≥ `max_wait_ms`. The timeout prevents latency spikes when only a few items are queued.
   - Enforces `max_inflight_batches` limit — selection phase pauses when too many batches are in-flight (backpressure: `max_pending = inflight_limit × batch_size`).
   - Tracks `ObjectRef → PendingNodeInfo` mapping for result correlation.

3. **RayInferenceClient** (`ray_client.py`): Sends async inference requests to Ray GPU actors:
   - **Least-loaded actor selection**: Picks actor with fewest pending requests (`_pending_counts[i]`) to distribute load evenly across GPU workers.
   - Returns `ray.ObjectRef` immediately (non-blocking) — caller proceeds while Ray handles compute.
   - The Ray actor internally runs an AsyncIO consumer loop with aggressive batching: blocks on first item, then non-blocking grabs until `batch_size` reached or deadline exceeded. Single `model(inputs)` forward pass over the concatenated batch, then splits results back to per-request futures.

**Virtual loss mechanism** (2-3 paragraphs)
- When a leaf is selected and enqueued, every node on the path from root to leaf gets `pending_visits += 1`.
- UCB calculation includes pending visits:

$$UCB = Q + c \cdot P \cdot \frac{\sqrt{N + N_{\text{pending}}}}{1 + N_{\text{child}} + N_{\text{child\_pending}}}$$

- This inflates the "visited" count, making that path less attractive for the next selection. Combined with the `inflight_to_root` dict (prevents selecting the exact same leaf), this naturally spreads selection across diverse paths.
- When results return: `pending_visits -= 1` on all path nodes. Error handling: rollback virtual loss on exception with `max(0, pending_visits - 1)` guard to prevent negative counts. A `finally` block in `search()` cleans up all outstanding virtual loss on early exit (KeyboardInterrupt, exception), preventing stale `pending_visits > 0` from corrupting future searches.

<!-- DIAGRAM: Pipelined async MCTS timeline. Show two parallel lanes: CPU (tree selection, enqueue, drain results) and GPU (batch inference). Show how they overlap: while GPU processes Batch 1, CPU selects leaves for Batch 2. Mark virtual loss application and removal. Show the three phases (Selection, Dispatch, Drain) in a loop. -->

**Code blocks (pick best 3):**
- `alphazero/gomoku/pvmcts/search/ray/ray_async.py:218-246` — virtual loss application + enqueue with error rollback (shows the careful engineering: apply virtual loss, enqueue, and if anything fails, rollback — this is what makes async MCTS correct, not just fast)
- `alphazero/gomoku/pvmcts/search/ray/batch_inference_manager.py:154-161` — `_should_dispatch()` (the batch_size OR timeout dispatch decision — simple but elegant adaptive batching)
- `alphazero/gomoku/inference/ray_client.py:289-315` — `infer_async()` with least-loaded actor selection (shows distributed systems thinking)

---

#### `04.training-pipeline.md` — Training Pipeline and GCP Infrastructure

**Section 1: Self-Play Loop** (2-3 paragraphs)
- Each iteration: the current champion model plays games against itself to generate training data. Each game produces a sequence of (state, MCTS_policy, outcome) tuples.
- Games are played with exploration: Dirichlet noise at root, temperature=1.0 for the first 20 moves (diverse opening play), then temperature→0 (deterministic).
- Opponent diversity: not 100% self-play. Mix of opponents prevents overfitting to a single play style. The ratio is configurable (e.g., 5% random bot, 30% previous champion, 65% current champion).

**Section 2: Replay Buffer and Data Augmentation** (2-3 paragraphs)
- Game records stored as Parquet shards (columnar format, efficient for large datasets with mixed types — board as int8 buffer, policy as float16).
- **D4 symmetry augmentation**: The 19×19 board has 8 symmetries (4 rotations × 2 reflections). Each game sample is augmented 8×, effectively multiplying the dataset. This is critical for data efficiency — a single game generates 8× the training signal.
- Replay buffer holds up to 500K samples with FIFO eviction. Minimum sample count required before training begins.

**Section 3: Prioritized Experience Replay (PER)** (2-3 paragraphs)
- After iteration 15, samples are weighted by prediction error. Positions where the model was most wrong are sampled more frequently:

$$P(i) = \frac{p_i^\alpha}{\sum_k p_k^\alpha}, \quad w_i = (N \cdot P(i))^{-\beta}$$

- Design requirement: PER only works if sample priorities are updated after each training step based on the prediction error from that step. Without this update, the priority weighting becomes stale and provides no benefit — it's just random sampling with extra overhead. This is a subtle but critical implementation detail in PER.

**Section 4: Training Loop** (2 paragraphs)
- 2 epochs per iteration (AlphaZero standard — prevents overfitting to the current replay buffer).
- Batch size 512, AMP (automatic mixed precision) for GPU efficiency, Adam optimizer with scheduled learning rate.
- Loss: MSE for value + cross-entropy for policy + L2 regularization.

**Section 5: Scheduled Parameters** (1-2 paragraphs)
- Many hyperparameters change over the course of training. The config system (Pydantic models) supports scheduled values:
```yaml
learning_rate:
  - { until: 30, value: 0.002 }
  - { until: 137, value: 0.0008 }
```
- This enables learning rate warmup/decay, increasing MCTS simulations as the model improves, and temperature annealing — all configured declaratively in YAML.

**Section 6: GCP Training Infrastructure** (2-3 paragraphs)
- Training ran on a GCP Ray cluster with GPU workers. Ray handles distributed self-play: multiple CPU workers generate games in parallel, sending inference requests to GPU actors.
- **Training progression observations**: The model started exhibiting blocking behavior (defending against opponent threats) around training day 3. As iterations progressed, it developed aggressive play patterns — including creating open threes (3-3 patterns). The current champion model plays recognizably strategic Gomoku.
- Data is stored in GCS and checkpoints are saved per iteration. The manifest (`manifest.json`) tracks training metadata, promotion history, and config snapshots for reproducibility.

<!-- DIAGRAM: D4 symmetry group. Show a small board (5×5 for clarity) and its 8 symmetry transformations: original, 90°, 180°, 270° rotations, plus horizontal flip of each. -->

**Code blocks (pick best 3):**
- `alphazero/gomoku/alphazero/learning/dataset.py` — D4 symmetry augmentation (the rotation + flip code — shows data engineering)
- `alphazero/gomoku/alphazero/learning/trainer.py` — `train_one_iteration` core loop (AMP, loss computation — shows ML engineering)
- `alphazero/configs/` — Config YAML snippet showing scheduled parameters (shows the config system design)

---

#### `05.evaluation.md` — Model Evaluation: Arena and SPRT

**Section 1: The Promotion Gate** (2-3 paragraphs)
- After training produces a new model (the "challenger"), it must prove it's stronger than the current best (the "champion") in head-to-head matches. If the challenger replaced the champion without verification and happened to be weaker, all subsequent self-play games would use that weaker model — generating lower-quality training data. This compounds: worse data → worse next model → even worse data. The arena evaluation prevents this regression loop.

**Section 2: Arena Evaluation** (2 paragraphs)
- 40 head-to-head games between challenger and champion, alternating who plays Black/White.
- Promotion threshold: 55% win rate. Evaluation uses deterministic play: temperature=0 (argmax), no Dirichlet noise, 600 MCTS simulations (3× training simulations for more accurate play).

**Section 3: SPRT** (2 paragraphs)
- For faster evaluation, SPRT (Sequential Probability Ratio Test) can terminate the match early once statistical significance is reached.
- Log-likelihood ratio:

$$LLR = n \cdot \left[ w \cdot \ln\frac{p_1}{p_0} + (1-w) \cdot \ln\frac{1-p_1}{1-p_0} \right]$$

  If LLR exceeds upper boundary: accept challenger (promote). If below lower boundary: reject. Otherwise: keep playing.

**Section 4: Elo Tracking** (1-2 paragraphs)
- Cumulative Elo tracked across iterations:

$$\Delta\text{Elo} = 400 \cdot \log_{10}\left(\frac{w}{1-w}\right)$$

- After 215 iterations, reached ~4160 cumulative Elo.

**Code blocks (pick best 2-3):**
- `alphazero/gomoku/alphazero/eval/arena.py` — arena match orchestration (shows the evaluation loop)
- `alphazero/gomoku/alphazero/eval/sprt.py` — SPRT log-likelihood ratio computation and decision logic

---

#### `06.cpp-extensions.md` — C++ Extensions: pybind11 for Performance

**Section 1: Why C++** (1-2 paragraphs)
- Two performance bottlenecks in Python: (1) double-three detection requires deep recursive rule checking, called for every legal move computation, and (2) MCTS tree operations (selection, expansion, backup) are called hundreds of times per move — Python loop overhead adds up.
- The Python implementation remains the default and serves as the reference. C++ is optional: `use_native: true` in config.

**Section 2: Two Modules** (2-3 paragraphs)
- **`renju_cpp`**: Low-level rules module. `CForbiddenPointFinder` implements double-three detection — sourced from [renju.se](https://www.renju.se/renlib/opensrc/), adapted from the minimax implementation. Direction-based open-three counting on a padded board.
- **`gomoku_cpp`**: High-level module. `GomokuCore` mirrors the Python game engine (state management, move application, capture detection, win checking, state encoding). `MctsEngine` is a native MCTS engine — selection, expansion, and backup happen in C++, with a Python callback for neural network inference.

**Section 3: Build System** (1 paragraph)
- scikit-build-core + CMake for C++ extensions. pybind11 3.0.1 for Python bindings. C++14 standard. Produces two importable `.so` modules.

**Section 4: Performance Impact** (1-2 paragraphs)
- Native MCTS provides approximately 5× speedup for the search phase.
- In production: 20 MCTS simulations with native C++ produce responses in under 1 second on a CPU VM. The server reconstructs native C++ state from the frontend WebSocket payload per request — no persistent native state between requests.

**Code blocks (pick best 2-3):**
- `alphazero/cpp/src/MctsEngine.cpp` — MctsEngine constructor and `search` method signature (shows the C++ MCTS interface)
- `alphazero/gomoku/pvmcts/search/sequential/cpp_strategy.py` — Python-side native MCTS invocation (shows the pybind11 bridge)

---

#### `07.deployment.md` — Production Deployment and Lessons Learned

**Section 1: Deployment Architecture** (2-3 paragraphs)
- GCP VMs running Container-Optimized OS (COS). Docker images built and pushed to GCP Artifact Registry.
- Two backend VMs: one for minimax, one for AlphaZero. Cloudflare Worker handles path-based routing (`/minimax/*` → minimax VM, `/alphazero/*` → AlphaZero VM).
- Infrastructure-as-scripts: `01_setup.sh` (one-time GCP project setup), `02_deploy.sh` (build + push + deploy), `03_deploy_cloudflare.sh` (DNS + Worker). No Terraform — uses `gcloud` CLI and Wrangler CLI directly.

**Section 2: Production Serving Configuration** (1-2 paragraphs)
- AlphaZero production Docker image (`Dockerfile.prod`): Python 3.13-slim, builds C++ extensions at image build time, torch-cpu.
- Production config (`deploy.yaml`): native C++ MCTS enabled, 20 simulations per move, CPU inference, cgroup-aware thread management (auto-detects container CPU limits).
- Stateless: each WebSocket request includes the full board state. Native C++ state is reconstructed from the payload, MCTS runs, and the result is returned.

**Section 3: Known Limitations and Room for Improvement** (2-3 paragraphs)
- The current model does not consistently achieve center opening play. In the AlphaZero paper, this strategic knowledge emerged from significantly more training iterations with higher compute budgets. The current limitation is a training budget constraint, not an architectural one — the system is capable of continued training.
- The model does demonstrate strong mid-game and defensive play: blocking opponent threats, creating capture opportunities, and executing aggressive 3-3 patterns.
- Potential technical improvements: ONNX export for faster CPU inference, model distillation to a smaller network for lower latency.

**Section 4: Lessons Learned** (3-4 paragraphs)

This is the most important section for demonstrating engineering maturity. Be honest, be specific.

- **Unit testing catches what gameplay testing can't.** The original implementation had data integrity issues — outcome values with wrong signs, PER priorities that were never updated — that were invisible during casual gameplay. The model would train, games would play, everything looked normal. But the training signal was corrupted. The complete refactoring that produced the current codebase started with building a proper test suite. This was the single most impactful change: once the tests existed, bugs that had been silently degrading training for weeks were caught immediately.

- **Fix the foundation before optimizing.** There are infinite optimization paths in an AlphaZero system: faster MCTS, better network architecture, curriculum learning, opening books. The temptation is to chase performance. But the biggest gains came from getting the fundamentals right — correct data pipeline, proper PER weight updates, reliable evaluation gating. A faster system that trains on corrupted data is worse than a slower system that trains correctly.

- **The gap between paper and implementation is real.** Reading the AlphaZero paper gives you the algorithm. It doesn't give you the engineering: how to handle distributed state synchronization, how to debug a model that plays but doesn't improve, how to manage GPU costs on a budget, how to detect when a rule change mid-training has contaminated your replay buffer. This gap is where the real learning happened.

- **More compute unlocks more capability, but diminishing returns require patience.** The model's progression was non-linear: long plateaus of seemingly no improvement, then sudden jumps in play quality (blocking patterns, aggressive tactics). Understanding that this is expected behavior of RL training — and having the discipline to keep running — is as important as the technical implementation.

**Code blocks (pick best 2-3):**
- `alphazero/configs/deploy.yaml` — production config (shows the serving configuration: use_native, num_searches, device)
- `alphazero/server/engine.py` — CPU thread management / cgroup detection (shows production-aware engineering)

---

## What NOT to Include
1. Full code listings — use focused 5-15 line snippets, max 3 per file
2. Basic CS concepts (hash maps, recursion, what Python is)
3. Build/install instructions (that's in "About Project" section)
4. Duplicate WebSocket protocol docs (reference the existing About Gomoku docs)
5. Raw training logs, Elo curves, or verbose config dumps — summarize results
6. Git/Docker/GCP tutorials — describe architecture and decisions, not how-to
7. Paper summaries — brief attribution + link, focus on what was implemented
8. Frontend documentation — left as placeholder per requirements
9. Explicit bug lists — frame as design decisions and engineering maturity, not error reports
10. Burnout, personal struggles, or time pressure — mention "compute budget constraint" or "time constraint" professionally if needed

## Tone Guide
- Practitioner explaining decisions, not textbook defining concepts
- "I chose X because Y" over "X is defined as..."
- When explaining fundamentals, do so in context of why they matter for THIS implementation
- Lead with the problem → explain the solution → show snippet or formula → note the tradeoff or lesson
- Be honest about limitations — engineers respect "this doesn't work yet because X" far more than "everything is great"

## Narrative Integration Points
- `minimax/00.intro.md`: Python → C++ migration motivation, C++98 personal choice from 42 experience
- `minimax/01.search-algorithms.md`: Progressive enhancement story, PVS with chessprogramming.org + LLM help
- `minimax/02.evaluation.md`: Bitmask double-three failed but evaluation bitmask still works — knowing what to keep
- `minimax/03.rules-and-serving.md`: CForbiddenPointFinder from renju.se replacing failed bitmask approach
- `alphazero/00.intro.md`: Papers → tutorial → refactoring arc (no explicit bug list, no timeframe)
- `alphazero/03.search-engine-architecture.md`: Async pipelining — the hardest engineering challenge, virtual loss, backpressure
- `alphazero/04.training-pipeline.md`: PER design requirement (framed as knowledge), training progression observations
- `alphazero/07.deployment.md`: Known limitations (center opening), lessons learned (testing, foundations, paper-to-implementation gap, RL patience)

## Verification
- After writing all files: run `cd front && npm run dev` and navigate to docs
- Check sidebar renders all files in correct order under Minimax and Alphazero accordion groups
- Verify KaTeX formulas render (inline `$...$` and block `$$...$$`)
- Verify code blocks have syntax highlighting (cpp, python)
- Check diagram placeholders have sufficient descriptions for later creation
- Verify live demo link (sungyongcho.com/gomoku) works
- Test all external hyperlinks (renju.se, chessprogramming.org, YouTube, GitHub)
