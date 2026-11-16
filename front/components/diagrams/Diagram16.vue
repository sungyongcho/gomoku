<template>
  <div class="diagram-wrapper">
    <div class="container">
      <h1>v5 Continuation Configs (Iterations 1–150)</h1>

      <div class="controls-panel">
        <div class="nav-row">
          <button type="button" class="nav-btn" :disabled="selectedIdx === 0" @click="selectedIdx--">
            ← Prev
          </button>
          <span class="stage-counter">Stage {{ selectedIdx + 1 }} / {{ STAGES.length }}</span>
          <button type="button" class="nav-btn" :disabled="selectedIdx === STAGES.length - 1" @click="selectedIdx++">
            Next →
          </button>
        </div>

        <div class="pill-row" role="group" aria-label="Continuation configs">
          <button
            v-for="(st, idx) in STAGES"
            :key="st.id"
            type="button"
            class="pill-btn"
            :class="{ active: selectedIdx === idx }"
            @click="selectedIdx = idx"
          >
            {{ st.short }}
          </button>
        </div>
      </div>

      <svg
        class="diagram-svg"
        viewBox="0 0 780 280"
        xmlns="http://www.w3.org/2000/svg"
        role="img"
        aria-label="Timeline of v5 continuation configs and learning-rate schedule over iterations 1 to 150"
      >
        <!-- Top section: Learning rate schedule + phase markers -->
        <text x="58" y="24" font-size="12.5" font-weight="700" fill="#0f172a">
          Planned Learning Rate &amp; Regime Changes Across Iterations 1–150
        </text>

        <!-- Plot frame -->
        <rect x="58" y="34" width="698" height="122" rx="4" fill="#ffffff" stroke="#e2e8f0" stroke-width="1" />

        <!-- Horizontal LR guides -->
        <g v-for="lvl in LR_LEVELS" :key="lvl.val">
          <line x1="58" :y1="lrToY(lvl.val)" x2="756" :y2="lrToY(lvl.val)" stroke="#f1f5f9" stroke-width="1" />
          <text x="52" :y="lrToY(lvl.val) + 4" text-anchor="end" font-size="10" fill="#64748b">
            {{ lvl.label }}
          </text>
        </g>

        <!-- Highlight active iteration range -->
        <rect
          :x="iterToX(activeStage.spanStart)"
          y="34"
          :width="Math.max(8, iterToX(activeStage.spanEnd) - iterToX(activeStage.spanStart))"
          height="122"
          fill="#eff6ff"
          opacity="0.9"
        />

        <!-- LR step polyline (actual path run through 150) -->
        <polyline
          :points="lrPolyline"
          fill="none"
          stroke="#2563eb"
          stroke-width="2.5"
          stroke-linejoin="round"
        />

        <!-- Key milestone vertical ticks -->
        <g v-for="m in MILESTONES" :key="m.iter">
          <line
            :x1="iterToX(m.iter)"
            y1="36"
            :x2="iterToX(m.iter)"
            y2="156"
            :stroke="m.color"
            stroke-width="1.4"
            stroke-dasharray="3 2"
          />
          <circle :cx="iterToX(m.iter)" cy="40" r="3" :fill="m.color" />
        </g>

        <!-- X axis iteration ticks -->
        <g v-for="it in [1, 20, 40, 60, 80, 100, 120, 144, 150]" :key="'xt-' + it">
          <line :x1="iterToX(it)" y1="156" :x2="iterToX(it)" y2="162" stroke="#64748b" stroke-width="1.2" />
          <text
            :x="iterToX(it)"
            y="174"
            text-anchor="middle"
            font-size="10.5"
            :font-weight="it === 144 ? '700' : '500'"
            :fill="it === 144 ? '#15803d' : '#475569'"
          >
            {{ it === 144 ? '144★' : it }}
          </text>
        </g>

        <!-- Clickable config blocks bar -->
        <text x="58" y="196" font-size="11.5" font-weight="700" fill="#475569">
          Config Timeline (click any block):
        </text>
        <g
          v-for="(st, idx) in STAGES"
          :key="'blk-' + st.id"
          class="stage-block"
          @click="selectedIdx = idx"
        >
          <rect
            :x="blockX(idx)"
            :y="st.row === 0 ? 204 : 238"
            :width="blockW(idx)"
            height="28"
            rx="5"
            :fill="selectedIdx === idx ? '#2563eb' : '#ffffff'"
            :stroke="selectedIdx === idx ? '#1e40af' : '#cbd5e1'"
            stroke-width="1.5"
          />
          <text
            :x="blockX(idx) + blockW(idx) / 2"
            :y="(st.row === 0 ? 204 : 238) + 18"
            text-anchor="middle"
            font-size="11.5"
            font-weight="700"
            :fill="selectedIdx === idx ? '#ffffff' : '#334155'"
          >
            {{ st.short }}
          </text>
        </g>
      </svg>

      <!-- Detail Card -->
      <div class="detail-card">
        <div class="detail-head">
          <span class="cfg-file">alphazero/configs/{{ activeStage.file }}</span>
          <span class="iter-badge">{{ activeStage.iters }}</span>
        </div>

        <div class="detail-grid">
          <div class="detail-box trigger">
            <div class="box-title">Trigger &amp; Measurement</div>
            <p>{{ activeStage.trigger }}</p>
          </div>
          <div class="detail-box changes">
            <div class="box-title">Config &amp; Code Changes</div>
            <ul>
              <li v-for="(c, i) in activeStage.changes" :key="i">{{ c }}</li>
            </ul>
          </div>
        </div>
      </div>

      <p class="source-note">
        All values, iteration spans, and trigger measurements are taken from the headers and schedules in <code>alphazero/configs/v5-small-6x96*.yaml</code>.
      </p>
    </div>
  </div>
</template>

<script setup>
import { computed, ref } from 'vue';

const STAGES = [
  {
    id: 'base',
    short: 'base (1–37)',
    file: 'v5-small-6x96.yaml',
    iters: 'Iters 1–37 (planned 50)',
    spanStart: 1,
    spanEnd: 37,
    row: 0,
    x: 58,
    w: 112,
    trigger:
      'Pilot 2026-11-07 (3 C4D + head): in-worker bf16 reached ~10,700 pos/s (240 games in 15 min) vs ~6,000 via Ray inference actors. FPU sweep 2026-11-07: r_fpu=0.0 left targets near uniform; 0.2+ narrowed to ~8 moves. From iter 29, reply_near (within 2 pts) rose from 2–4% to 53–68%.',
    changes: [
      '6x96 ResNet from random init, use_local_inference: true (bfloat16)',
      'num_searches: 400, fpu_reduction: 0.1, exploration_turns: 8',
      'lr: 0.001, replay_buffer_size: 250k, num_epochs: 2',
      'Promo gate: 40 games every 2 iters, 4 random opening plies',
    ],
  },
  {
    id: 'cont',
    short: 'cont (37–51)',
    file: 'v5-small-6x96-cont.yaml',
    iters: 'Switched 2026-11-08 after iter 36/38',
    spanStart: 37,
    spanEnd: 51,
    row: 0,
    x: 174,
    w: 84,
    trigger:
      'Two promotion evals in a row failed. Against v4 (12x128, 400 searches), iter 24 lost 4–36 and iter 28 lost 6–34: v4 built double threats a few moves ahead.',
    changes: [
      'num_searches & eval_num_searches raised 400 → 800',
      'LR schedule: 0.001 (≤50), 0.0005 (≤75), 0.0002 (≤100)',
      'Head workers: 12 → 13 (num_actors: 0 freed 1 head vCPU)',
      'Added 40-game v4 reference match every 10 iters (at 400 searches)',
    ],
  },
  {
    id: 'cont2',
    short: 'cont2 (52–61)',
    file: 'v5-small-6x96-cont2.yaml',
    iters: 'From iter 52',
    spanStart: 52,
    spanEnd: 61,
    row: 0,
    x: 262,
    w: 76,
    trigger:
      'At iter 50, 96% of self-play games opened on (5,2) & mirrors with almost no stones ≥ 7 pts from the edge. Against v4 from random openings, v5 lost 27/40 (median 10 plies).',
    changes: [
      'random_opening_turns cycles 1, 2, 3, 4, 0 from iter 52 (unsearched, excluded from targets)',
      'exploration_turns raised 8 → 16',
      'lr: 0.001 extended through iter 80 (then 0.0005 ≤92, 0.0002 ≤100)',
    ],
  },
  {
    id: 'cont3',
    short: 'cont3 (62–100)',
    file: 'v5-small-6x96-cont3.yaml',
    iters: 'Iters 62–100',
    spanStart: 62,
    spanEnd: 100,
    row: 0,
    x: 342,
    w: 112,
    trigger:
      'At iter 56, v5 returned to its home corner and lost 40/40 to v4 from a center opening, valuing an immediate win at −0.60. Local 1-epoch test: center 34/40 with translation vs 0/40 without; random 26/40 vs 19/40.',
    changes: [
      'Added translation_prob: 0.5 (shifts ~25% of samples without outer-line stones)',
      'Dropped policy mass in wrapped edge strip zeroed and renormalized',
    ],
  },
  {
    id: 'cont4',
    short: 'cont4 (101–110)',
    file: 'v5-small-6x96-cont4.yaml',
    iters: 'Iters 101–110',
    spanStart: 101,
    spanEnd: 110,
    row: 0,
    x: 458,
    w: 96,
    trigger:
      'Iters 64–100 loss fell 1.86 → 1.32 while strength stayed flat (49.6% over 480 promo games; 0.725–0.875 vs v4 cluster; 0.75–0.80 locally; 8/9 local losses were unblocked outer-line fives, 0.2% of replay).',
    changes: [
      '25% self-play games vs v4 (400 searches vs v5 800; both sides recorded)',
      'LR bumped to 0.0005 (101–112 planned; switched early at 111)',
      'num_eval_games doubled 40 → 80 (at 40 games, equal models pass 55% gate 32% of the time)',
    ],
  },
  {
    id: 'cont5',
    short: 'cont5 (111–130)',
    file: 'v5-small-6x96-cont5.yaml',
    iters: 'Iters 111–130',
    spanStart: 111,
    spanEnd: 130,
    row: 0,
    x: 558,
    w: 98,
    trigger:
      'cont4 lifted v4 random-opening score to 0.95, yet blocking an outer-line five stayed at 0.62–0.63 top-1 because Ray self-play only drew root noise on move 0.',
    changes: [
      'Code fix (baefedb): root noise on every searched move, zero-prior children created, random openings for both sides',
      'lr lowered to 0.0002 from iter 111',
      'Dropped 58-min cluster v4 match (12 min locally); logged self-play score vs external_bot',
    ],
  },
  {
    id: 'cont6',
    short: 'cont6 (131–150)',
    file: 'v5-small-6x96-cont6.yaml',
    iters: 'From iter 131',
    spanStart: 131,
    spanEnd: 150,
    row: 1,
    x: 58,
    w: 165,
    trigger:
      'Strength plateaued since iter 84 (110 scored 53.75%, 116 scored 55% vs 84; 118–130 scored 47.5–52.5% vs 116). v5 now beat v4 39–1 at equal search, making v4 moves (1/8 of targets) harmful.',
    changes: [
      'Pure self-play restored (v4 external ratio → 0.0)',
      'lr: 0.0001 (131–145), 0.00005 (146–150)',
      'replay_buffer_size: 250k → 500k with num_epochs: 1 (16 iters of games seen once)',
    ],
  },
  {
    id: 'cont7',
    short: 'cont7 (+ladder)',
    file: 'v5-small-6x96-cont7.yaml',
    iters: 'During iters 131–150',
    spanStart: 131,
    spanEnd: 135,
    row: 1,
    x: 230,
    w: 170,
    trigger:
      'Promotion gate only tested latest vs one champion, and v4 no longer separated late v5 models.',
    changes: [
      'Added round-robin ladder every 5 iters: latest, 5-iter weight average, anchor 116, anchor 130',
      '10 opening pairs (20 games/pairing), 800 searches, 4-ply opening, max 400 plies',
    ],
  },
  {
    id: 'cont8',
    short: 'cont8 (3-ply gate)',
    file: 'v5-small-6x96-cont8.yaml',
    iters: 'After iter 135',
    spanStart: 135,
    spanEnd: 140,
    row: 1,
    x: 407,
    w: 172,
    trigger:
      'Iter 135 ladder scored 9–11/20 everywhere and gate 40–40 due to 4-ply Black bias (Black won 87%/66%/79%/38%/82% for 0–4 plies over 1200 games/iter in 131–135). 12/120 ladder games took 20–48 min.',
    changes: [
      'Switched gate & ladder to 3 random opening plies (White moves first, 38% Black win rate)',
      'Ladder max_plies: 80, pairs: 20 (40 games/pairing)',
    ],
  },
  {
    id: 'cont9',
    short: 'cont9 (drop ladder)',
    file: 'v5-small-6x96-cont9.yaml',
    iters: 'After iter 140',
    spanStart: 140,
    spanEnd: 150,
    row: 1,
    x: 586,
    w: 170,
    trigger:
      'Ladders at 135 (9–11/20) and 140 (19–21/40) held the cluster 48–62 min because ~281k denormal weights slowed float32 Intel C4 nodes to 206–258 ms/pos vs 7 ms AMD (self-play bf16: 5 ms).',
    changes: [
      'Dropped in-run ladder; kept 3-ply promotion gate',
      'Final checkpoint selection moved post-run to scripts/ladder.py (flushes denormals → picked iter 144)',
    ],
  },
];

const LR_LEVELS = [
  { val: 0.001, label: '1e-3' },
  { val: 0.0005, label: '5e-4' },
  { val: 0.0002, label: '2e-4' },
  { val: 0.0001, label: '1e-4' },
  { val: 0.00005, label: '5e-5' },
];

// Actual LR schedule executed across iterations 1-150 (from cont6/cont9 schedule comments)
const LR_SEGMENTS = [
  [1, 0.001],
  [80, 0.001],
  [80, 0.0005],
  [92, 0.0005],
  [92, 0.0002],
  [100, 0.0002],
  [100, 0.0005],
  [110, 0.0005],
  [110, 0.0002],
  [130, 0.0002],
  [130, 0.0001],
  [145, 0.0001],
  [145, 0.00005],
  [150, 0.00005],
];

const MILESTONES = [
  { iter: 37, color: '#64748b' },
  { iter: 52, color: '#d97706' },
  { iter: 62, color: '#2563eb' },
  { iter: 101, color: '#7c3aed' },
  { iter: 111, color: '#dc2626' },
  { iter: 131, color: '#0891b2' },
  { iter: 144, color: '#16a34a' },
];

const selectedIdx = ref(0);
const activeStage = computed(() => STAGES[selectedIdx.value]);

function iterToX(it) {
  return 58 + ((it - 1) / 149) * 698;
}

function lrToY(lr) {
  // Log scale from 4e-5 to 1.2e-3
  const lo = Math.log10(0.00004);
  const hi = Math.log10(0.0012);
  const norm = (Math.log10(lr) - lo) / (hi - lo);
  return 150 - norm * 106;
}

const lrPolyline = computed(() =>
  LR_SEGMENTS.map(([it, lr]) => `${iterToX(it).toFixed(1)},${lrToY(lr).toFixed(1)}`).join(' ')
);

function blockX(idx) {
  return STAGES[idx].x;
}

function blockW(idx) {
  return STAGES[idx].w;
}
</script>

<style scoped>
*, *::before, *::after {
  box-sizing: border-box;
}

.diagram-wrapper {
  font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
  background: #f8fafc;
  color: #1e293b;
  display: flex;
  justify-content: center;
  padding: 18px 14px;
  width: 100%;
  max-width: 100%;
  overflow-x: hidden;
}

.container {
  width: 100%;
  max-width: 920px;
  margin: 0 auto;
}

h1 {
  font-size: 19px;
  font-weight: 700;
  text-align: center;
  margin: 0 0 12px 0;
  color: #0f172a;
  letter-spacing: -0.2px;
}

.controls-panel {
  display: flex;
  flex-direction: column;
  gap: 8px;
  margin-bottom: 10px;
}

.nav-row,
.pill-row {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: center;
  gap: 6px;
}

.nav-btn,
.pill-btn {
  font-family: inherit;
  font-size: 12px;
  font-weight: 600;
  padding: 5px 10px;
  border: 1.5px solid #cbd5e1;
  border-radius: 6px;
  background: #ffffff;
  color: #334155;
  cursor: pointer;
  transition: all 0.15s ease;
}

.nav-btn:hover:not(:disabled),
.pill-btn:hover {
  background: #f1f5f9;
  border-color: #94a3b8;
}

.nav-btn:disabled {
  opacity: 0.4;
  cursor: default;
}

.pill-btn.active {
  background: #eff6ff;
  border-color: #2563eb;
  color: #1e40af;
}

.stage-counter {
  font-size: 12.5px;
  font-weight: 600;
  color: #475569;
  padding: 0 6px;
}

.diagram-svg {
  width: 100%;
  max-width: 100%;
  height: auto;
  display: block;
}

.stage-block {
  cursor: pointer;
}

.detail-card {
  background: #ffffff;
  border: 1.5px solid #cbd5e1;
  border-radius: 8px;
  padding: 12px 14px;
  margin-top: 10px;
}

.detail-head {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  margin-bottom: 10px;
  padding-bottom: 8px;
  border-bottom: 1px solid #e2e8f0;
}

.cfg-file {
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
  font-size: 12.5px;
  font-weight: 700;
  color: #1e3a8a;
}

.iter-badge {
  font-size: 12px;
  font-weight: 700;
  background: #eff6ff;
  color: #1e40af;
  border: 1px solid #bfdbfe;
  border-radius: 999px;
  padding: 2px 10px;
}

.detail-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 12px;
}

.detail-box {
  background: #f8fafc;
  border: 1px solid #e2e8f0;
  border-radius: 6px;
  padding: 10px 12px;
  font-size: 12.5px;
  line-height: 1.5;
  color: #334155;
}

.detail-box.trigger {
  border-left: 3px solid #d97706;
}

.detail-box.changes {
  border-left: 3px solid #2563eb;
}

.box-title {
  font-size: 12px;
  font-weight: 700;
  color: #0f172a;
  margin-bottom: 5px;
}

.detail-box p {
  margin: 0;
}

.detail-box ul {
  margin: 0;
  padding-left: 17px;
}

.detail-box li {
  margin-bottom: 3px;
}

.source-note {
  margin: 8px 4px 0 4px;
  font-size: 12px;
  line-height: 1.45;
  color: #64748b;
  text-align: center;
}

@media (max-width: 640px) {
  .detail-grid {
    grid-template-columns: 1fr;
  }
}

@media (max-width: 480px) {
  h1 {
    font-size: 16.5px;
  }
}
</style>
