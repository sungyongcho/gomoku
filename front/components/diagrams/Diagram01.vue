<template>
  <div class="diagram-wrapper">
    <div class="container">
      <h1>AlphaZero Training and Serving Loop</h1>
      <p class="hint">Click a stage for details, or step through one iteration. The gate outcome decides which path lights up.</p>

      <div class="controls">
        <div class="group">
          <button type="button" class="pick" @click="step(-1)">Back</button>
          <button type="button" :class="['pick', 'play', { active: playing }]" @click="togglePlay">
            {{ playing ? 'Pause' : 'Play the loop' }}
          </button>
          <button type="button" class="pick" @click="step(1)">Next step</button>
        </div>
        <div class="group" role="radiogroup" aria-label="Gate outcome">
          <span class="group-label">Gate outcome</span>
          <button
            v-for="o in OUTCOMES"
            :key="o.id"
            type="button"
            role="radio"
            :aria-checked="outcome === o.id"
            :class="['pick', 'outcome', o.id, { active: outcome === o.id }]"
            @click="setOutcome(o.id)"
          >{{ o.label }}</button>
        </div>
      </div>

      <div class="loop">
        <button
          v-for="s in STAGE_LIST"
          :key="s.id"
          type="button"
          :class="['card', s.id, { active: selected === s.id, dim: !onPath(s.id) }]"
          :style="{ gridArea: s.id }"
          :aria-pressed="selected === s.id"
          @click="pick(s.id)"
        >
          <span class="card-step">{{ s.step }}</span>
          <span class="card-title">{{ s.title }}</span>
          <span v-for="(l, i) in s.lines" :key="i" class="card-line">{{ l }}</span>
          <span v-if="s.mobileNote" class="card-note">{{ s.mobileNote }}</span>
        </button>

        <div
          v-for="a in ARROWS"
          :key="a.id"
          :class="['arr', a.dir, a.id, { lit: litArrows.includes(a.id) }]"
          :style="{ gridArea: a.id }"
          aria-hidden="true"
        >
          <span v-if="a.label" class="arr-label">{{ a.label }}</span>
        </div>
        <div :class="['elbow', { lit: litArrows.includes('aca') }]" aria-hidden="true" />
      </div>

      <section ref="detailEl" :class="['detail', selected]" aria-live="polite">
        <p class="detail-step">{{ current.step }}</p>
        <h2>{{ current.title }}</h2>
        <ul>
          <li v-for="(f, i) in current.facts" :key="i">
            <template v-for="(part, j) in f" :key="j">
              <code v-if="j % 2 === 1">{{ part }}</code>
              <template v-else>{{ part }}</template>
            </template>
          </li>
        </ul>
        <p class="src">
          <span>Source (<code>before-optimization</code>):</span>
          <a
            v-for="l in current.src.map(srcLink)"
            :key="l.title"
            :href="l.href"
            :title="l.title"
            target="_blank"
            rel="noopener"
          ><code>{{ l.text }}</code></a>
        </p>
      </section>
    </div>
  </div>
</template>

<script setup>
import { computed, onBeforeUnmount, ref } from 'vue';

// Facts are split on backticks: odd parts render as code.
const f = (s) => s.split('`');

// Sources are repo paths on the before-optimization branch, as "path:line".
const G = 'alphazero/gomoku/';
const RUN = G + 'scripts/pipelines/run_ray.py';
const COMMON = G + 'scripts/pipelines/common.py';
const BLOB = 'https://github.com/sungyongcho/gomoku/blob/before-optimization/';
const srcLink = (r) => {
  const [file, line] = r.split(':');
  return { href: `${BLOB}${file}#L${line}`, text: `${file.split('/').pop()}:${line}`, title: r };
};

const STAGES = {
  selfplay: {
    step: 'Step 1',
    title: 'Self-play',
    lines: ['latest network vs itself', 'sampled openings → (s, π, z)'],
    facts: [
      f('Workers get the weights the trainer just produced, not the champion. A restarted run first loads the champion checkpoint.'),
      f('Self-play asks for root Dirichlet noise (`add_noise=True`), but the batched native search V4 used never mixed it in, so V4 games ran without it (see MCTS). Moves are sampled from π at the scheduled temperature (0.75–1.30 in V4) for the first `exploration_turns` (20–30), then played greedily.'),
      f('Every position becomes a sample: state s, MCTS policy π, and the final outcome z from the mover’s side.'),
      f('Opponent mix: some games face a random checkpoint from the last five iterations (`prev_bot_ratio` 0.3 at the V4 start, 0 from iteration 31, 0.25 from the day-6 recovery) or a random bot.'),
      f('V4 schedule: 3000 → 1200 games per iteration, 200 → 2400 searches per move.'),
    ],
    src: [RUN + ':580', G + 'alphazero/runners/workers/ray_worker.py:368', G + 'alphazero/runners/selfplay.py:76', COMMON + ':467'],
  },
  buffer: {
    step: 'Step 2',
    title: 'Replay buffer',
    lines: ['FIFO · 250K–500K samples', 'random D4 view per load'],
    facts: [
      f('A bounded deque with `maxlen = training.replay_buffer_size`: the oldest samples fall out first.'),
      f('V4 sizes: 500K at the start, later 400K, 350K, 450K and 250K.'),
      f('Each iteration also writes its samples to a Parquet shard; a restarted run warm-starts the buffer from those shards.'),
      f('One of the 8 D4 symmetries (4 rotations × 2 reflections) is picked at random each time a sample is loaded.'),
    ],
    src: [COMMON + ':145', RUN + ':588', G + 'alphazero/learning/dataset.py:63'],
  },
  training: {
    step: 'Step 3',
    title: 'Training',
    lines: ['2 epochs · batch 512', 'Adam + AMP · CE + MSE'],
    facts: [
      f('Trains on the whole buffer every iteration: 2 epochs, batch 512 in every V4 config.'),
      f('Loss = policy cross-entropy against π + value MSE against z.'),
      f('Adam with weight decay 1e-4; float16 autocast with a GradScaler (AMP) on the GPU.'),
      f('The learning rate follows a per-iteration schedule: 0.002 early in V4, 0.0006–0.001 later.'),
      f('The result is saved as this iteration’s checkpoint: the challenger.'),
    ],
    src: [RUN + ':737', G + 'alphazero/learning/trainer.py:184', G + 'scripts/train.py:65', RUN + ':749'],
  },
  arena: {
    step: 'Step 4',
    title: 'Arena',
    lines: ['challenger vs champion', '40 games · no root noise'],
    facts: [
      f('Runs on every 2nd iteration in V4 (`eval_every_iters: 2`); the other iterations go straight to the next one.'),
      f('40 games. The Ray evaluator splits them across workers, and each worker starts with the challenger as Black; at one game per worker, the challenger played Black in every V4 game.'),
      f('Searches run without root noise. From the day-6 recovery on, the first 6 moves were sampled at temperature 1.0; before that, play was greedy from move 1.'),
      f('Arena searches per move: 600, later 800, then 1000.'),
    ],
    src: [RUN + ':253', G + 'alphazero/runners/ray_eval_runner.py:151', G + 'alphazero/eval/arena.py:146', G + 'alphazero/eval/match.py:87'],
  },
  gate: {
    step: 'Step 5',
    title: 'Promotion gate',
    lines: ['score ≥ 55%?', 'rejected: champion stays'],
    facts: [
      f('Promote when the score reaches `promotion_win_rate` (0.55), where score = (wins + ½ draws) / games.'),
      f('The Ray pipeline marks the baseline and blunder checks as passed, so the score alone decides.'),
      f('SPRT early stopping is off (`use_sprt: false`), so all 40 games are played.'),
      f('Rejected: the champion stays, and the rejected weights keep training in the next iteration.'),
    ],
    src: [RUN + ':303', RUN + ':346', RUN + ':357'],
  },
  next: {
    step: 'Loop',
    title: 'Next iteration',
    lines: ['keeps the latest network', 'promoted or not'],
    mobileNote: '↺ back to step 1',
    facts: [
      f('Self-play reads `model.state_dict()`: the weights just trained, whether or not they were promoted.'),
      f('Iterations without an arena come straight here after training.'),
      f('The champion only decides the arena opponent, the restart point and the deployed model.'),
    ],
    src: [RUN + ':580', RUN + ':253'],
  },
  champion: {
    step: 'On promotion',
    title: 'Champion checkpoint',
    lines: ['arena opponent', 'restart point · deployed'],
    mobileNote: '↺ next arena opponent',
    facts: [
      f('Starts as the initial network, saved as iteration 0 (randomly initialized for V4).'),
      f('Arena opponent in every evaluation until a challenger beats it.'),
      f('Its path is kept in `manifest.json` (`champion_model_path`); a restarted run resumes from it.'),
      f('`02_deploy.sh` copies the manifest’s champion into the serving image as `models/champion.pt`.'),
    ],
    src: [G + 'scripts/train.py:149', RUN + ':268', COMMON + ':97', 'deploy/alphazero/02_deploy.sh:80'],
  },
  serving: {
    step: 'Deployed',
    title: 'Serving',
    lines: ['200 sims · native C++', 'stateless per request'],
    facts: [
      f('FastAPI + WebSocket server; loads `models/champion.pt` with `configs/deploy.yaml`.'),
      f('200 MCTS simulations per move with the native C++ engine (`use_native: true`).'),
      f('No root noise and no exploration: it plays the top move of the root policy.'),
      f('Stateless: each request carries the full board, and every move builds a fresh search tree.'),
    ],
    src: ['alphazero/server/__init__.py:15', 'alphazero/configs/deploy.yaml:21', 'alphazero/server/engine.py:85'],
  },
};

const STAGE_LIST = Object.entries(STAGES).map(([id, s]) => ({ id, ...s }));

const ARROWS = [
  { id: 'a1', dir: 'r' },
  { id: 'a2', dir: 'r' },
  { id: 'a3', dir: 'd', label: 'challenger' },
  { id: 'a4', dir: 'l' },
  { id: 'agn', dir: 'l', label: 'either way' },
  { id: 'anx', dir: 'u', label: 'latest weights' },
  { id: 'agc', dir: 'd', label: 'promoted' },
  { id: 'aca', dir: 'u', label: 'opponent' },
  { id: 'acs', dir: 'l', label: 'deploy' },
];

const OUTCOMES = [
  { id: 'promote', label: 'Promoted' },
  { id: 'reject', label: 'Rejected' },
];

const PATHS = {
  promote: ['selfplay', 'buffer', 'training', 'arena', 'gate', 'champion', 'serving', 'next'],
  reject: ['selfplay', 'buffer', 'training', 'arena', 'gate', 'next'],
};

// Arrows that carry data into each stage.
const LIT = {
  selfplay: ['anx'],
  buffer: ['a1'],
  training: ['a2'],
  arena: ['a3', 'aca'],
  gate: ['a4'],
  champion: ['agc'],
  serving: ['acs'],
  next: ['agn'],
};

const selected = ref('selfplay');
const detailEl = ref(null);
const outcome = ref('promote');
const playing = ref(false);
let timer = null;

const current = computed(() => STAGES[selected.value]);
const litArrows = computed(() => LIT[selected.value]);
const path = computed(() => PATHS[outcome.value]);
// The champion stays the arena opponent on either path, so it never dims.
const onPath = (id) => id === 'champion' || path.value.includes(id);

function step(dir, fromTimer = false) {
  if (!fromTimer) stop();
  const p = path.value;
  let i = p.indexOf(selected.value);
  if (i === -1) i = p.indexOf('gate');
  let n = i + dir;
  if (n >= p.length) {
    n = 0;
    // Auto-play alternates outcomes so both return paths get shown.
    if (fromTimer) outcome.value = outcome.value === 'promote' ? 'reject' : 'promote';
  }
  if (n < 0) n = p.length - 1;
  selected.value = p[n];
}

function stop() {
  playing.value = false;
  if (timer) clearInterval(timer);
  timer = null;
}

function togglePlay() {
  if (playing.value) return stop();
  playing.value = true;
  timer = setInterval(() => step(1, true), 1700);
}

function pick(id) {
  stop();
  if (id === 'champion' || id === 'serving') outcome.value = 'promote';
  selected.value = id;
  // On narrow screens the panel sits far below the cards; bring it into view.
  const el = detailEl.value;
  if (el && el.getBoundingClientRect().top > window.innerHeight - 80) {
    el.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
  }
}

function setOutcome(id) {
  outcome.value = id;
}

onBeforeUnmount(stop);
</script>

<style scoped>
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

.diagram-wrapper {
  font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
  background: #f8fafc;
  color: #1e293b;
  padding: 24px 16px;
  container-type: inline-size;
}
.container { width: 100%; max-width: 1000px; margin: 0 auto; }

h1 { font-size: 19px; font-weight: 700; text-align: center; color: #0f172a; }
.hint { font-size: 13px; color: #64748b; text-align: center; margin: 4px 0 12px; }

.controls {
  display: flex; flex-wrap: wrap; justify-content: center; align-items: center;
  gap: 10px 22px; margin-bottom: 18px;
}
.group { display: flex; flex-wrap: wrap; justify-content: center; align-items: center; gap: 8px; }
.group-label { font-size: 12px; font-weight: 700; color: #64748b; text-transform: uppercase; letter-spacing: 0.04em; }
.pick {
  font-family: inherit; font-size: 13px; font-weight: 600;
  padding: 7px 14px; border-radius: 999px; cursor: pointer;
  border: 1.5px solid #cbd5e1; background: #fff; color: #334155;
  transition: background 0.15s, border-color 0.15s;
}
.pick:hover { border-color: #94a3b8; }
.pick.play.active { background: #dbeafe; border-color: #2563eb; color: #1e3a8a; }
.pick.outcome.promote.active { background: #ede9fe; border-color: #7c3aed; color: #4c1d95; }
.pick.outcome.reject.active { background: #fee2e2; border-color: #dc2626; color: #7f1d1d; }

/* Loop grid: three card columns joined by arrow columns. */
.loop {
  --gap: 68px;
  display: grid;
  grid-template-columns: minmax(0, 1fr) var(--gap) minmax(0, 1fr) var(--gap) minmax(0, 1fr);
  grid-template-rows: auto 44px auto 44px auto;
  grid-template-areas:
    "selfplay a1  buffer a2    training"
    "anx      .   .      .     a3"
    "next     agn gate   a4    arena"
    ".        .   agc    .     aca"
    "serving  acs champion elbow elbow";
}

.card {
  --c: #64748b; --bg: #fff;
  display: flex; flex-direction: column; align-items: flex-start; text-align: left;
  min-width: 0; padding: 10px 12px; border-radius: 10px;
  border: 1.5px solid var(--c); background: var(--bg);
  font-family: inherit; color: inherit; cursor: pointer;
  transition: box-shadow 0.2s, opacity 0.2s, transform 0.2s;
}
.card:hover { box-shadow: 0 0 0 3px rgba(148, 163, 184, 0.35); }
.card.active { border-width: 2.5px; padding: 9px 11px; box-shadow: 0 0 0 4px color-mix(in srgb, var(--c) 22%, transparent); }
.card.dim:not(.active) { opacity: 0.45; }
.card.selfplay { --c: #2563eb; --bg: #eff6ff; }
.card.buffer { --c: #0891b2; --bg: #ecfeff; }
.card.training { --c: #059669; --bg: #ecfdf5; }
.card.arena { --c: #d97706; --bg: #fffbeb; }
.card.gate { --c: #b45309; --bg: #fef3c7; }
.card.next { --c: #64748b; --bg: #fff; border-style: dashed; }
.card.next.active { border-style: solid; }
.card.champion { --c: #7c3aed; --bg: #f5f3ff; }
.card.serving { --c: #475569; --bg: #f1f5f9; }

.card-step { font-size: 11px; font-weight: 700; letter-spacing: 0.04em; text-transform: uppercase; color: var(--c); }
.card-title { font-size: 15px; font-weight: 700; color: #0f172a; margin: 1px 0 4px; }
.card-line { font-size: 12.5px; line-height: 1.45; color: #334155; }
.card-note { display: none; font-size: 12px; font-weight: 600; color: var(--c); margin-top: 4px; }

/* Arrows */
.arr { --c: #94a3b8; position: relative; min-width: 0; }
.arr.lit, .elbow.lit { --c: #2563eb; }
.arr::before, .arr::after, .elbow::before, .elbow::after { content: ''; position: absolute; }
.arr::before { background: var(--c); }
.arr.r::before, .arr.l::before { top: calc(50% - 1px); height: 2px; left: 4px; right: 4px; }
.arr.d::before, .arr.u::before { left: calc(50% - 1px); width: 2px; top: 4px; bottom: 4px; }
.arr.r::after { top: calc(50% - 6px); right: 0; border: 6px solid transparent; border-left: 8px solid var(--c); border-right: 0; }
.arr.l::after { top: calc(50% - 6px); left: 0; border: 6px solid transparent; border-right: 8px solid var(--c); border-left: 0; }
.arr.d::after { left: calc(50% - 6px); bottom: 0; border: 6px solid transparent; border-top: 8px solid var(--c); border-bottom: 0; }
.arr.u::after { left: calc(50% - 6px); top: 0; border: 6px solid transparent; border-bottom: 8px solid var(--c); border-top: 0; }
.arr.aca::before { bottom: 0; }
.arr.lit::before { box-shadow: 0 0 0 0.5px var(--c); }

.arr-label {
  position: absolute; font-size: 11px; font-weight: 600; color: #64748b;
  white-space: nowrap; line-height: 1;
}
.arr.lit .arr-label { color: #1d4ed8; }
.arr.d .arr-label, .arr.u .arr-label { left: calc(50% + 9px); top: 50%; transform: translateY(-50%); }
.arr.l .arr-label, .arr.r .arr-label { left: 50%; bottom: calc(50% + 6px); transform: translateX(-50%); }

/* Champion -> arena: right from the champion card, then up into arena. */
.elbow { grid-area: elbow; position: relative; --c: #94a3b8; }
.elbow::before { top: calc(50% - 1px); height: 2px; left: 4px; right: calc((100% - var(--gap)) / 2 - 1px); background: var(--c); }
.elbow::after { top: 0; height: calc(50% + 1px); width: 2px; right: calc((100% - var(--gap)) / 2 - 1px); background: var(--c); }

/* Detail panel */
.detail {
  --c: #64748b;
  margin-top: 18px; background: #fff; border: 1.5px solid #e2e8f0;
  border-left: 5px solid var(--c); border-radius: 10px; padding: 12px 16px;
}
.detail.selfplay { --c: #2563eb; }
.detail.buffer { --c: #0891b2; }
.detail.training { --c: #059669; }
.detail.arena { --c: #d97706; }
.detail.gate { --c: #b45309; }
.detail.champion { --c: #7c3aed; }
.detail.serving { --c: #475569; }
.detail-step { font-size: 11px; font-weight: 700; letter-spacing: 0.04em; text-transform: uppercase; color: var(--c); }
.detail h2 { font-size: 15px; font-weight: 700; color: #0f172a; margin: 1px 0 8px; }
.detail ul { padding-left: 18px; }
.detail li { font-size: 13px; line-height: 1.55; list-style: disc; margin-bottom: 3px; }
.src {
  display: flex; flex-wrap: wrap; align-items: baseline; gap: 4px 8px;
  margin-top: 10px; padding-top: 8px; border-top: 1px dashed #cbd5e1;
  font-size: 11.5px; color: #64748b;
}
.src a { text-decoration: none; }
.src a code { font-size: 11px; color: #1d4ed8; }
.src a:hover code { text-decoration: underline; }
.src > span code { font-size: 11px; background: none; padding: 0; }

code {
  font-family: 'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 0.88em; background: rgba(15, 23, 42, 0.06); padding: 0.05em 0.3em; border-radius: 4px;
  overflow-wrap: anywhere;
  font-variant-ligatures: none;
}

/* Phone: main loop runs top to bottom, then the gate splits into two columns. */
@container (max-width: 620px) {
  .loop {
    grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
    grid-template-rows: none;
    grid-auto-rows: auto;
    column-gap: 10px;
    grid-template-areas:
      "selfplay selfplay"
      "a1       a1"
      "buffer   buffer"
      "a2       a2"
      "training training"
      "a3       a3"
      "arena    arena"
      "a4       a4"
      "gate     gate"
      "agn      agc"
      "next     champion"
      ".        acs"
      ".        serving";
  }
  .anx, .aca, .elbow { display: none; }
  .arr { height: 30px; }
  .arr.r::before, .arr.l::before { top: 4px; bottom: 4px; height: auto; left: calc(50% - 1px); right: auto; width: 2px; }
  .arr.r::after, .arr.l::after {
    top: auto; bottom: 0; left: calc(50% - 6px); right: auto;
    border: 6px solid transparent; border-top: 8px solid var(--c); border-bottom: 0;
  }
  .arr.l .arr-label, .arr.r .arr-label { left: calc(50% + 9px); bottom: auto; top: 50%; transform: translateY(-50%); }
  .card-note { display: block; }
  .card-title { font-size: 14px; }
  .card-line { font-size: 12px; }
}
</style>
