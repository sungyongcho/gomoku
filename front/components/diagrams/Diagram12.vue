<template>
  <div class="diagram-wrapper">
    <div class="container">
      <h1>First-Play Urgency (FPU) in PUCT Selection</h1>

      <div class="controls-panel">
        <div class="sim-controls">
          <button type="button" class="ctrl-btn" :disabled="simCount === 0" @click="resetSims">↺ Reset (0)</button>
          <button type="button" class="ctrl-btn" :disabled="simCount >= MAX_SIMS" @click="stepSims(1)">+1 Sim</button>
          <button type="button" class="ctrl-btn" :disabled="simCount >= MAX_SIMS" @click="stepSims(5)">+5 Sims</button>
          <button
            type="button"
            class="ctrl-btn"
            :class="{ playing: isPlaying }"
            @click="toggleAuto"
          >
            {{ isPlaying ? '⏸ Pause' : '▶ Auto' }}
          </button>
          <span class="sim-badge">Sims: <strong>{{ simCount }}</strong> / {{ MAX_SIMS }}</span>
        </div>

        <div class="param-controls">
          <div class="param-item">
            <label for="fpu-slider">
              <code>fpu_reduction</code>: <strong>{{ fpuReduction.toFixed(2) }}</strong>
            </label>
            <input
              id="fpu-slider"
              v-model.number="fpuReduction"
              type="range"
              min="0"
              max="0.8"
              step="0.05"
              class="range-input"
            />
            <div class="preset-btns">
              <button
                v-for="val in [0.0, 0.1, 0.2, 0.4, 0.6]"
                :key="val"
                type="button"
                class="preset-btn"
                :class="{ active: Math.abs(fpuReduction - val) < 0.001 }"
                @click="fpuReduction = val"
              >
                {{ val.toFixed(1) }}
              </button>
            </div>
          </div>

          <div class="param-item">
            <label for="vparent-slider">
              Parent value <code>V_parent</code>: <strong>{{ fmtSigned(parentValue, 1) }}</strong>
            </label>
            <input
              id="vparent-slider"
              v-model.number="parentValue"
              type="range"
              min="-0.8"
              max="0.8"
              step="0.1"
              class="range-input"
            />
          </div>
        </div>

        <div class="priors-controls">
          <span class="priors-title">Move weights (normalized to priors <code>P</code>):</span>
          <div class="priors-grid">
            <label v-for="(w, idx) in weights" :key="idx" class="prior-cell">
              <span>M{{ idx + 1 }}: {{ priors[idx].toFixed(2) }}</span>
              <input
                v-model.number="weights[idx]"
                type="range"
                min="0.02"
                max="0.60"
                step="0.02"
                class="prior-slider"
              />
            </label>
          </div>
          <button type="button" class="preset-btn" @click="resetPriors">Reset priors</button>
        </div>
      </div>

      <svg
        class="diagram-svg"
        viewBox="0 0 780 420"
        xmlns="http://www.w3.org/2000/svg"
        role="img"
        aria-label="Interactive PUCT selection and FPU reduction illustration with a flat toy evaluator"
      >
        <!-- Toy disclaimer banner -->
        <rect x="18" y="8" width="744" height="26" rx="6" fill="#fffbeb" stroke="#fcd34d" stroke-width="1.2" />
        <text x="390" y="25" text-anchor="middle" font-size="12" font-weight="600" fill="#92400e">
          Illustration with a toy flat evaluator (every leaf returns V_parent) — not a measurement
        </text>

        <!-- Root Node -->
        <rect x="210" y="44" width="360" height="68" rx="8" fill="#eff6ff" stroke="#2563eb" stroke-width="2" />
        <text x="390" y="65" text-anchor="middle" font-size="14" font-weight="700" fill="#1e3a8a">
          Root · N_parent = {{ state.rootVisits }} (1 + {{ simCount }} sims)
        </text>
        <text x="390" y="84" text-anchor="middle" font-size="12.5" fill="#1e40af">
          W/N = {{ fmtSigned(state.rootMean, 2) }} · fpu_reduction = {{ fpuReduction.toFixed(2) }} · c_puct = 2.0
        </text>
        <text x="390" y="102" text-anchor="middle" font-size="12" fill="#475569">
          √(N_parent + 1) = √{{ state.rootVisits + 1 }} = {{ Math.sqrt(state.rootVisits + 1).toFixed(3) }}
        </text>

        <!-- Edges from Root to Children -->
        <g v-for="(child, idx) in state.children" :key="'edge-' + idx">
          <line
            x1="390"
            y1="112"
            :x2="cardCenterX(idx)"
            y2="148"
            :stroke="edgeColor(idx)"
            :stroke-width="edgeWidth(idx)"
            :stroke-dasharray="idx === state.nextIdx && idx !== state.lastPickedIdx ? '5 3' : undefined"
          />
        </g>

        <!-- Child Cards -->
        <g v-for="(child, idx) in state.children" :key="'card-' + idx">
          <rect
            :x="cardX(idx)"
            y="148"
            width="114"
            height="190"
            rx="8"
            :fill="cardFill(idx)"
            :stroke="cardStroke(idx)"
            :stroke-width="idx === state.nextIdx || idx === state.lastPickedIdx ? '2.2' : '1.5'"
          />

          <!-- Title + status tag -->
          <text :x="cardCenterX(idx)" y="169" text-anchor="middle" font-size="14" font-weight="700" fill="#0f172a">
            M{{ idx + 1 }}
          </text>
          <text
            v-if="idx === state.nextIdx"
            :x="cardCenterX(idx)"
            y="184"
            text-anchor="middle"
            font-size="10.5"
            font-weight="700"
            fill="#2563eb"
          >
            next pick
          </text>
          <text
            v-else-if="idx === state.lastPickedIdx"
            :x="cardCenterX(idx)"
            y="184"
            text-anchor="middle"
            font-size="10.5"
            font-weight="700"
            fill="#15803d"
          >
            just visited
          </text>
          <text
            v-else
            :x="cardCenterX(idx)"
            y="184"
            text-anchor="middle"
            font-size="10.5"
            fill="#64748b"
          >
            {{ child.visits === 0 ? 'unvisited' : 'visited' }}
          </text>

          <!-- Prior -->
          <text :x="cardCenterX(idx)" y="203" text-anchor="middle" font-size="12" font-weight="600" fill="#334155">
            P = {{ child.prior.toFixed(3) }}
          </text>

          <!-- Visit share bar -->
          <rect :x="cardX(idx) + 11" y="212" width="92" height="10" rx="4" fill="#e2e8f0" />
          <rect
            v-if="child.visits > 0 && simCount > 0"
            :x="cardX(idx) + 11"
            y="212"
            :width="(child.visits / Math.max(1, simCount)) * 92"
            height="10"
            rx="4"
            :fill="idx === state.lastPickedIdx ? '#16a34a' : '#2563eb'"
          />

          <!-- Visit count -->
          <text :x="cardCenterX(idx)" y="241" text-anchor="middle" font-size="15" font-weight="700" fill="#0f172a">
            N = {{ child.visits }}
          </text>

          <!-- Q value -->
          <text
            :x="cardCenterX(idx)"
            y="262"
            text-anchor="middle"
            font-size="12"
            :font-weight="child.visits === 0 ? '700' : '500'"
            :fill="child.visits === 0 ? '#b45309' : '#334155'"
          >
            Q = {{ fmtSigned(child.q, 3) }}
          </text>
          <text
            :x="cardCenterX(idx)"
            y="276"
            text-anchor="middle"
            font-size="10.5"
            :fill="child.visits === 0 ? '#b45309' : '#64748b'"
          >
            {{ child.visits === 0 ? '(V_p − r_fpu)' : '(−W_c / N_c)' }}
          </text>

          <!-- Exploration term U -->
          <text :x="cardCenterX(idx)" y="297" text-anchor="middle" font-size="12" fill="#334155">
            U = +{{ child.u.toFixed(3) }}
          </text>

          <!-- Total UCB -->
          <line :x1="cardX(idx) + 12" y1="306" :x2="cardX(idx) + 102" y2="306" stroke="#cbd5e1" stroke-width="1" />
          <text :x="cardCenterX(idx)" y="324" text-anchor="middle" font-size="12.5" font-weight="700" fill="#0f172a">
            UCB {{ fmtSigned(child.ucb, 3) }}
          </text>
        </g>

        <!-- Bottom summary bar -->
        <rect x="18" y="352" width="744" height="58" rx="8" fill="#ffffff" stroke="#cbd5e1" stroke-width="1.5" />
        <text x="390" y="374" text-anchor="middle" font-size="13" font-weight="700" fill="#0f172a">
          Visited moves: {{ visitedCount }} / 6  ·  Unvisited (N=0): {{ 6 - visitedCount }}  ·  Max visits: {{ maxVisits }}  ·  Top-2 share: {{ top2Share }}%
        </text>
        <text x="390" y="396" text-anchor="middle" font-size="11.5" fill="#475569">
          UCB = Q + 2.0 · P · √(N_parent + 1) / (1 + N_child)  ·  visited Q = −W_c / N_c  ·  unvisited Q = V_parent − r_fpu
        </text>
      </svg>
    </div>
  </div>
</template>

<script setup>
import { computed, onBeforeUnmount, ref } from 'vue';

const DEFAULT_WEIGHTS = [0.36, 0.24, 0.16, 0.11, 0.08, 0.05];
const C_PUCT = 2.0;
const MAX_SIMS = 60;

const weights = ref([...DEFAULT_WEIGHTS]);
const fpuReduction = ref(0.0);
const parentValue = ref(0.0);
const simCount = ref(18);
const isPlaying = ref(false);
let autoTimer = null;

const priors = computed(() => {
  const sum = weights.value.reduce((acc, w) => acc + w, 0);
  return weights.value.map((w) => (sum > 0 ? w / sum : 1 / weights.value.length));
});

const state = computed(() => {
  const p = priors.value;
  const vParent = parentValue.value;
  const rFpu = fpuReduction.value;

  let rootVisits = 1;
  let rootValueSum = vParent;
  const childVisits = p.map(() => 0);
  const childValueSum = p.map(() => 0);
  let lastPickedIdx = -1;

  const scoreChildren = () => {
    const parentMean = rootValueSum / rootVisits;
    const sqrtTerm = Math.sqrt(rootVisits + 1);
    return p.map((prior, idx) => {
      const n = childVisits[idx];
      const q = n > 0 ? -(childValueSum[idx] / n) : parentMean - rFpu;
      const u = (C_PUCT * prior * sqrtTerm) / (1 + n);
      return {
        prior,
        visits: n,
        q,
        u,
        ucb: q + u,
      };
    });
  };

  for (let s = 0; s < simCount.value; s++) {
    const scored = scoreChildren();
    let bestIdx = 0;
    for (let i = 1; i < scored.length; i++) {
      if (scored[i].ucb > scored[bestIdx].ucb + 1e-12) {
        bestIdx = i;
      }
    }
    // Flat evaluator: leaf from opponent's perspective returns -vParent
    childVisits[bestIdx] += 1;
    childValueSum[bestIdx] += -vParent;
    rootVisits += 1;
    rootValueSum += vParent;
    lastPickedIdx = bestIdx;
  }

  const finalScored = scoreChildren();
  let nextIdx = 0;
  for (let i = 1; i < finalScored.length; i++) {
    if (finalScored[i].ucb > finalScored[nextIdx].ucb + 1e-12) {
      nextIdx = i;
    }
  }

  return {
    rootVisits,
    rootMean: rootValueSum / rootVisits,
    children: finalScored,
    lastPickedIdx,
    nextIdx,
  };
});

const visitedCount = computed(
  () => state.value.children.filter((c) => c.visits > 0).length
);

const maxVisits = computed(() =>
  Math.max(0, ...state.value.children.map((c) => c.visits))
);

const top2Share = computed(() => {
  if (simCount.value === 0) return 0;
  const sorted = [...state.value.children]
    .map((c) => c.visits)
    .sort((a, b) => b - a);
  return Math.round(((sorted[0] + sorted[1]) / simCount.value) * 100);
});

function stopAuto() {
  if (autoTimer) {
    clearInterval(autoTimer);
    autoTimer = null;
  }
  isPlaying.value = false;
}

function toggleAuto() {
  if (isPlaying.value) {
    stopAuto();
    return;
  }
  if (simCount.value >= MAX_SIMS) {
    simCount.value = 0;
  }
  isPlaying.value = true;
  autoTimer = setInterval(() => {
    if (simCount.value >= MAX_SIMS) {
      stopAuto();
      return;
    }
    simCount.value += 1;
  }, 450);
}

function stepSims(delta) {
  stopAuto();
  simCount.value = Math.min(MAX_SIMS, simCount.value + delta);
}

function resetSims() {
  stopAuto();
  simCount.value = 0;
}

function resetPriors() {
  weights.value = [...DEFAULT_WEIGHTS];
}

function fmtSigned(val, decimals) {
  const fixed = val.toFixed(decimals);
  return val >= 0 ? `+${fixed}` : fixed;
}

function cardX(idx) {
  return 18 + idx * 126;
}

function cardCenterX(idx) {
  return cardX(idx) + 57;
}

function cardFill(idx) {
  if (idx === state.value.lastPickedIdx) return '#f0fdf4';
  if (idx === state.value.nextIdx) return '#eff6ff';
  return state.value.children[idx].visits === 0 ? '#fffbeb' : '#ffffff';
}

function cardStroke(idx) {
  if (idx === state.value.lastPickedIdx) return '#16a34a';
  if (idx === state.value.nextIdx) return '#2563eb';
  return state.value.children[idx].visits === 0 ? '#fcd34d' : '#cbd5e1';
}

function edgeColor(idx) {
  if (idx === state.value.lastPickedIdx) return '#16a34a';
  if (idx === state.value.nextIdx) return '#2563eb';
  return '#cbd5e1';
}

function edgeWidth(idx) {
  if (idx === state.value.lastPickedIdx || idx === state.value.nextIdx) return '2.2';
  return '1.3';
}

onBeforeUnmount(() => {
  stopAuto();
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
  display: flex;
  flex-direction: column;
  gap: 10px;
  margin-bottom: 10px;
}

.sim-controls,
.param-controls,
.priors-controls {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: center;
  gap: 8px;
}

.param-controls {
  gap: 14px;
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

.ctrl-btn,
.preset-btn {
  font-family: inherit;
  font-size: 12.5px;
  font-weight: 600;
  padding: 5px 11px;
  border: 1.5px solid #cbd5e1;
  border-radius: 6px;
  background: #ffffff;
  color: #334155;
  cursor: pointer;
  transition: all 0.15s ease;
}

.ctrl-btn:hover:not(:disabled),
.preset-btn:hover {
  background: #f1f5f9;
  border-color: #94a3b8;
}

.ctrl-btn:disabled {
  opacity: 0.4;
  cursor: default;
}

.ctrl-btn.playing {
  background: #fef3c7;
  border-color: #d97706;
  color: #92400e;
}

.preset-btn.active {
  background: #eff6ff;
  border-color: #2563eb;
  color: #1e40af;
}

.sim-badge {
  font-size: 12.5px;
  color: #475569;
  padding: 4px 8px;
  background: #ffffff;
  border: 1px solid #e2e8f0;
  border-radius: 6px;
}

.range-input {
  width: 135px;
  max-width: 100%;
  accent-color: #2563eb;
  cursor: pointer;
}

.preset-btns {
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
}

.priors-controls {
  background: #ffffff;
  border: 1px solid #e2e8f0;
  border-radius: 8px;
  padding: 8px 12px;
}

.priors-title {
  font-size: 12px;
  font-weight: 600;
  color: #475569;
}

.priors-grid {
  display: flex;
  flex-wrap: wrap;
  justify-content: center;
  gap: 8px 12px;
}

.prior-cell {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 2px;
  font-size: 11.5px;
  color: #334155;
}

.prior-slider {
  width: 72px;
  accent-color: #2563eb;
  cursor: pointer;
}

.diagram-svg {
  width: 100%;
  max-width: 100%;
  height: auto;
  display: block;
}

@media (max-width: 480px) {
  h1 {
    font-size: 16.5px;
  }
}
</style>
