<template>
  <div class="diagram-wrapper">
    <div class="container">
      <h1>In-Flight Inference Batches &amp; Root Visit Flattening</h1>

      <div class="controls-panel">
        <div class="param-row">
          <div class="param-item">
            <label for="inflight-slider">
              <code>max_inflight_batches</code>: <strong>{{ maxInflight }}</strong>
            </label>
            <input
              id="inflight-slider"
              v-model.number="maxInflight"
              type="range"
              min="1"
              max="25"
              step="1"
              class="range-input"
            />
            <div class="preset-btns">
              <button
                v-for="val in [1, 2, 4, 8, 25]"
                :key="val"
                type="button"
                class="preset-btn"
                :class="{ active: maxInflight === val }"
                @click="maxInflight = val"
              >
                {{ val === 1 ? '1 (cap)' : val === 25 ? '25 (uncapped)' : val }}
              </button>
            </div>
          </div>

          <div class="param-item">
            <span class="param-label">Evaluator latency:</span>
            <div class="preset-btns" role="group" aria-label="Evaluator latency">
              <button
                type="button"
                class="preset-btn"
                :class="{ active: delayRounds === 1 }"
                @click="delayRounds = 1"
              >
                Fast (1 poll)
              </button>
              <button
                type="button"
                class="preset-btn"
                :class="{ active: delayRounds === 20 }"
                @click="delayRounds = 20"
              >
                Slow remote (20 polls)
              </button>
            </div>
          </div>
        </div>
      </div>

      <svg
        class="diagram-svg"
        viewBox="0 0 780 415"
        xmlns="http://www.w3.org/2000/svg"
        role="img"
        aria-label="Illustrative simulation of how uncapped in-flight batches with virtual loss flatten root visit counts under slow inference"
      >
        <!-- Illustrative disclaimer banner -->
        <rect x="18" y="8" width="744" height="26" rx="6" fill="#fffbeb" stroke="#fcd34d" stroke-width="1.2" />
        <text x="390" y="25" text-anchor="middle" font-size="11.5" font-weight="600" fill="#92400e">
          Illustrative simulation (9×9 = 81 moves, 200 sims, batch 8, peaked logits, mirroring native_async_inflight_test.py)
        </text>

        <!-- Summary KPI cards -->
        <g v-for="(kpi, idx) in kpis" :key="kpi.label">
          <rect
            :x="18 + idx * 188"
            y="44"
            width="178"
            height="56"
            rx="8"
            :fill="kpi.fill"
            :stroke="kpi.stroke"
            stroke-width="1.4"
          />
          <text :x="107 + idx * 188" y="64" text-anchor="middle" font-size="11" font-weight="600" fill="#475569">
            {{ kpi.label }}
          </text>
          <text :x="107 + idx * 188" y="87" text-anchor="middle" font-size="15.5" font-weight="700" :fill="kpi.color">
            {{ kpi.value }}
          </text>
        </g>

        <!-- Main Chart Panel -->
        <rect x="18" y="112" width="744" height="240" rx="8" fill="#ffffff" stroke="#cbd5e1" stroke-width="1.4" />
        <text x="34" y="135" font-size="13" font-weight="700" fill="#0f172a">
          Root Visit Distribution (Top 10 + Tail of 81 Moves)
        </text>
        <text x="746" y="135" text-anchor="end" font-size="11.5" fill="#64748b">
          Peak in flight: {{ simResult.peakInflight }} batch{{ simResult.peakInflight === 1 ? '' : 'es' }} · {{ simResult.peakPending }} leaves
        </text>

        <!-- Horizontal reference line for 104 (sync M1 count) -->
        <line
          x1="56"
          :y1="barBaselineY - (104 / 110) * barMaxH"
          x2="585"
          :y2="barBaselineY - (104 / 110) * barMaxH"
          stroke="#94a3b8"
          stroke-width="1.2"
          stroke-dasharray="4 3"
        />
        <text x="585" :y="barBaselineY - (104 / 110) * barMaxH - 4" text-anchor="end" font-size="10.5" fill="#64748b">
          sync top-move visits (104)
        </text>

        <!-- Y-axis baseline -->
        <line x1="48" :y1="barBaselineY" x2="594" :y2="barBaselineY" stroke="#94a3b8" stroke-width="1.5" />

        <!-- Top 10 Move Bars -->
        <g v-for="(v, idx) in simResult.top10" :key="'bar-' + idx">
          <rect
            :x="62 + idx * 52"
            :y="barBaselineY - barHeight(v)"
            width="36"
            :height="barHeight(v)"
            rx="4"
            :fill="idx === 0 ? topMoveColor : idx < 3 ? '#60a5fa' : '#94a3b8'"
          />
          <text
            :x="80 + idx * 52"
            :y="barBaselineY - barHeight(v) - 6"
            text-anchor="middle"
            font-size="11.5"
            font-weight="700"
            fill="#0f172a"
          >
            {{ v }}
          </text>
          <text
            :x="80 + idx * 52"
            :y="barBaselineY + 16"
            text-anchor="middle"
            font-size="11"
            font-weight="600"
            fill="#334155"
          >
            {{ idx === 0 ? 'M1' : idx === 1 ? 'M2' : idx === 2 ? 'M3' : `M${idx + 1}` }}
          </text>
          <text
            :x="80 + idx * 52"
            :y="barBaselineY + 30"
            text-anchor="middle"
            font-size="9.5"
            fill="#64748b"
          >
            {{ idx === 0 ? '4.0' : idx < 3 ? '2.0' : '0.0' }}
          </text>
        </g>

        <text x="34" :y="barBaselineY + 30" font-size="9.5" fill="#64748b">logit:</text>

        <!-- Tail bucket panel (Moves 11..81) -->
        <rect x="610" y="154" width="138" height="178" rx="8" fill="#f8fafc" stroke="#e2e8f0" stroke-width="1.2" />
        <text x="679" y="176" text-anchor="middle" font-size="12" font-weight="700" fill="#1e293b">
          Other 71 Moves
        </text>
        <text x="679" y="192" text-anchor="middle" font-size="10.5" fill="#64748b">
          (M11–M81, logit 0.0)
        </text>

        <text x="679" y="222" text-anchor="middle" font-size="20" font-weight="700" :fill="simResult.tailVisited > 25 ? '#dc2626' : '#2563eb'">
          {{ simResult.tailVisited }} / 71
        </text>
        <text x="679" y="239" text-anchor="middle" font-size="10.5" fill="#475569">
          moves visited
        </text>

        <line x1="626" y1="254" x2="732" y2="254" stroke="#cbd5e1" stroke-width="1" />

        <text x="679" y="276" text-anchor="middle" font-size="15" font-weight="700" fill="#0f172a">
          {{ simResult.tailVisits }} visits
        </text>
        <text x="679" y="294" text-anchor="middle" font-size="10.5" fill="#475569">
          ({{ Math.round((simResult.tailVisits / 200) * 100) }}% of search budget)
        </text>
        <text x="679" y="318" text-anchor="middle" font-size="10" font-weight="600" :fill="simResult.width === 30 ? '#15803d' : '#b45309'">
          {{ simResult.width === 30 ? 'Matches sync search' : 'Flattened by pending VL' }}
        </text>

        <!-- Bottom explanation box -->
        <rect x="18" y="362" width="744" height="44" rx="6" fill="#ffffff" stroke="#cbd5e1" stroke-width="1.2" />
        <text x="390" y="381" text-anchor="middle" font-size="11.5" font-weight="600" fill="#1e293b">
          {{ explanationLine1 }}
        </text>
        <text x="390" y="398" text-anchor="middle" font-size="11" fill="#475569">
          {{ explanationLine2 }}
        </text>
      </svg>

      <p class="source-note">
        Illustrative model implementing <code>MctsEngine::CalculateUcb</code> and <code>RunAsyncSimulation</code> with <code>c_puct=2.0</code>, <code>fpu_reduction=0.1</code>, <code>batch_size=8</code>, <code>200</code> simulations, and neutral values.
      </p>
    </div>
  </div>
</template>

<script setup>
import { computed, ref } from 'vue';

const NUM_MOVES = 81;
const TOTAL_SIMS = 200;
const BATCH_SIZE = 8;
const C_PUCT = 2.0;
const FPU_REDUCTION = 0.1;

const PRIORS = (() => {
  const logits = new Array(NUM_MOVES).fill(0);
  logits[0] = 4.0;
  logits[1] = 2.0;
  logits[2] = 2.0;
  const exps = logits.map(Math.exp);
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / sum);
})();

const maxInflight = ref(1);
const delayRounds = ref(20);

const barBaselineY = 304;
const barMaxH = 136;

function barHeight(visits) {
  return Math.max(visits > 0 ? 4 : 0, (visits / 110) * barMaxH);
}

const simResult = computed(() => {
  const cap = maxInflight.value;
  const delay = delayRounds.value;

  const makeNode = (parent) => ({
    parent,
    n: 0,
    w: 0,
    vl: 0,
    expanded: false,
    children: null,
  });

  const root = makeNode(null);
  root.n = 1;
  root.expanded = true;
  root.children = PRIORS.map(() => makeNode(root));

  const ucb = (parent, child, prior) => {
    const pv = parent.n + parent.vl;
    const cv = child.n + child.vl;
    const q = child.n > 0 ? -(child.w + child.vl) / cv : parent.w / parent.n - FPU_REDUCTION;
    const u = (C_PUCT * prior * Math.sqrt(pv + 1)) / (1 + cv);
    return q + u;
  };

  let inFlight = [];
  let cur = [];
  let completed = 0;
  let dispatched = 0;
  let peakInflight = 0;
  let peakPending = 0;
  let guard = 0;

  while (completed < TOTAL_SIMS && guard++ < 10000) {
    let madeProgress = false;
    while (dispatched < TOTAL_SIMS && inFlight.length < cap && cur.length < BATCH_SIZE) {
      let node = root;
      const path = [root];
      while (node.expanded) {
        let bestIdx = 0;
        let bestScore = -Infinity;
        for (let i = 0; i < NUM_MOVES; i++) {
          const s = ucb(node, node.children[i], PRIORS[i]);
          if (s > bestScore) {
            bestScore = s;
            bestIdx = i;
          }
        }
        node = node.children[bestIdx];
        path.push(node);
      }
      const dup =
        cur.some((it) => it.leaf === node) ||
        inFlight.some((b) => b.items.some((it) => it.leaf === node));
      if (dup) break;
      for (const p of path) p.vl += 1;
      cur.push({ leaf: node, path });
      dispatched += 1;
      madeProgress = true;
    }

    if (cur.length > 0) {
      const shouldDispatch = cur.length >= BATCH_SIZE || dispatched >= TOTAL_SIMS || !madeProgress;
      if (shouldDispatch) {
        inFlight.push({ rem: delay, items: cur });
        cur = [];
        peakInflight = Math.max(peakInflight, inFlight.length);
        const pendingNow = inFlight.reduce((acc, b) => acc + b.items.length, 0);
        peakPending = Math.max(peakPending, pendingNow);
      }
    }

    if (inFlight.length > 0) {
      for (const b of inFlight) b.rem -= 1;
      const ready = inFlight.filter((b) => b.rem <= 0);
      inFlight = inFlight.filter((b) => b.rem > 0);
      for (const b of ready) {
        for (const it of b.items) {
          for (const p of it.path) p.vl -= 1;
          it.leaf.expanded = true;
          it.leaf.children = PRIORS.map(() => makeNode(it.leaf));
          let val = 0.0;
          for (let i = it.path.length - 1; i >= 0; i--) {
            it.path[i].n += 1;
            it.path[i].w += val;
            val = -val;
          }
          completed += 1;
        }
      }
    }
  }

  const visits = root.children.map((c) => c.n);
  const width = visits.filter((v) => v > 0).length;
  const top10 = visits.slice(0, 10);
  const tail = visits.slice(10);
  const tailVisits = tail.reduce((a, b) => a + b, 0);
  const tailVisited = tail.filter((v) => v > 0).length;
  const top3Share = Math.round(((visits[0] + visits[1] + visits[2]) / TOTAL_SIMS) * 100);

  return {
    top10,
    width,
    maxVisits: visits[0],
    tailVisits,
    tailVisited,
    top3Share,
    peakInflight,
    peakPending,
  };
});

const topMoveColor = computed(() => {
  if (simResult.value.width <= 32) return '#16a34a';
  if (simResult.value.width <= 60) return '#2563eb';
  return '#dc2626';
});

const kpis = computed(() => [
  {
    label: 'Top Move (M1) Visits',
    value: `${simResult.value.maxVisits} / 200`,
    fill: simResult.value.maxVisits >= 90 ? '#f0fdf4' : simResult.value.maxVisits >= 45 ? '#eff6ff' : '#fef2f2',
    stroke: simResult.value.maxVisits >= 90 ? '#86efac' : simResult.value.maxVisits >= 45 ? '#93c5fd' : '#fca5a5',
    color: simResult.value.maxVisits >= 90 ? '#166534' : simResult.value.maxVisits >= 45 ? '#1e40af' : '#991b1b',
  },
  {
    label: 'Visited Root Moves (Width)',
    value: `${simResult.value.width} / 81`,
    fill: simResult.value.width <= 32 ? '#f0fdf4' : simResult.value.width <= 60 ? '#fffbeb' : '#fef2f2',
    stroke: simResult.value.width <= 32 ? '#86efac' : simResult.value.width <= 60 ? '#fcd34d' : '#fca5a5',
    color: simResult.value.width <= 32 ? '#166534' : simResult.value.width <= 60 ? '#92400e' : '#991b1b',
  },
  {
    label: 'Top-3 Moves Share',
    value: `${simResult.value.top3Share}%`,
    fill: '#ffffff',
    stroke: '#cbd5e1',
    color: '#0f172a',
  },
  {
    label: 'Peak Batches In Flight',
    value: `${simResult.value.peakInflight} / ${maxInflight.value}`,
    fill: '#ffffff',
    stroke: '#cbd5e1',
    color: '#0f172a',
  },
]);

const explanationLine1 = computed(() => {
  if (delayRounds.value === 1) {
    return 'With fast inference (1 poll), batches drain immediately so raising max_inflight_batches does not build a backlog.';
  }
  if (maxInflight.value === 1) {
    return 'With max_inflight_batches = 1, selection waits after each batch, expands M1, and concentrates 104 visits on M1 (width 30).';
  }
  return `With slow inference and max_inflight_batches = ${maxInflight.value}, ${simResult.value.peakPending} pending leaves hold virtual loss at once.`;
});

const explanationLine2 = computed(() => {
  if (delayRounds.value === 1) {
    return 'Switch to Slow remote (20 polls) to see how delayed GPU replies interact with virtual loss.';
  }
  if (maxInflight.value === 1) {
    return 'Matches synchronous batched MCTS. Drag the slider right to watch root visits flatten across all 81 moves.';
  }
  return 'Each virtual loss lowers UCB on M1–M3 before any values return, pushing unvisited low-prior moves to be selected instead.';
});
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
  margin-bottom: 10px;
}

.param-row {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: center;
  gap: 10px 18px;
}

.param-item {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: center;
  gap: 8px;
  font-size: 12.5px;
  color: #334155;
}

.param-label {
  font-weight: 600;
  color: #475569;
}

.range-input {
  width: 150px;
  max-width: 100%;
  accent-color: #2563eb;
  cursor: pointer;
}

.preset-btns {
  display: flex;
  flex-wrap: wrap;
  gap: 5px;
}

.preset-btn {
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

.preset-btn:hover {
  background: #f1f5f9;
  border-color: #94a3b8;
}

.preset-btn.active {
  background: #eff6ff;
  border-color: #2563eb;
  color: #1e40af;
}

.diagram-svg {
  width: 100%;
  max-width: 100%;
  height: auto;
  display: block;
}

.source-note {
  margin: 8px 4px 0 4px;
  font-size: 12px;
  line-height: 1.45;
  color: #64748b;
  text-align: center;
}

@media (max-width: 480px) {
  h1 {
    font-size: 16.5px;
  }
  .range-input {
    width: 100%;
  }
}
</style>
