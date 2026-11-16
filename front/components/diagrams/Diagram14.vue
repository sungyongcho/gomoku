<template>
  <div class="diagram-wrapper">
    <div class="container">
      <h1>Root Dirichlet Noise Mixer &amp; Zero-Prior Child Creation</h1>

      <div class="controls-panel">
        <div class="slider-group">
          <div class="slider-item">
            <label for="eps-slider" class="slider-label">
              Noise weight <code>ε</code>: <strong>{{ epsilon.toFixed(2) }}</strong>
            </label>
            <input
              id="eps-slider"
              v-model.number="epsilon"
              type="range"
              min="0"
              max="0.5"
              step="0.05"
              class="range-input"
            />
            <div class="preset-btns">
              <button
                type="button"
                class="preset-btn"
                :class="{ active: isClose(epsilon, 0.0) }"
                @click="epsilon = 0.0"
              >
                0.00 (eval)
              </button>
              <button
                type="button"
                class="preset-btn"
                :class="{ active: isClose(epsilon, 0.25) }"
                @click="epsilon = 0.25"
              >
                0.25 (v5)
              </button>
            </div>
          </div>

          <div class="slider-item">
            <label for="alpha-slider" class="slider-label">
              Concentration <code>α</code>: <strong>{{ alpha.toFixed(2) }}</strong>
            </label>
            <input
              id="alpha-slider"
              v-model.number="alpha"
              type="range"
              min="0.01"
              max="0.5"
              step="0.01"
              class="range-input"
            />
            <div class="preset-btns">
              <button
                type="button"
                class="preset-btn"
                :class="{ active: isClose(alpha, 0.03) }"
                @click="resampleWithAlpha(0.03)"
              >
                0.03 (v5)
              </button>
              <button
                type="button"
                class="preset-btn"
                :class="{ active: isClose(alpha, 0.15) }"
                @click="resampleWithAlpha(0.15)"
              >
                0.15 (v4)
              </button>
            </div>
          </div>
        </div>

        <div class="action-row">
          <button type="button" class="action-btn primary" @click="resampleNoise">
            Resample η ~ Dir({{ alpha.toFixed(2) }})
          </button>
          <button type="button" class="action-btn" @click="spikeOnZeroPrior">
            Force η spike on M6 (P = 0)
          </button>
          <span class="draw-badge">Draw #{{ drawCount }}</span>
        </div>
      </div>

      <div class="status-bar" :class="m6Mixed > 0 ? 'is-rescued' : 'is-zero'">
        <strong>Zero-prior move M6 (P = 0.000):</strong>
        <template v-if="epsilon === 0">
          With <code>ε = 0.00</code>, no noise is mixed in; <code>P'(M6) = 0.000</code> and no child is created.
        </template>
        <template v-else-if="m6Mixed > 0.0005">
          Noise <code>η₆ = {{ noise[5].toFixed(3) }}</code> gives <code>P'(M6) = {{ m6Mixed.toFixed(3) }}</code>.
          <code>_ensure_root_noise</code> creates a <code>TreeNode</code> for every nonzero entry so M6 becomes searchable.
        </template>
        <template v-else>
          Small <code>α = {{ alpha.toFixed(2) }}</code> concentrates <code>η</code> on just one or two moves, so this draw gave <code>η₆ ≈ 0</code>. Resample or click "Force η spike on M6".
        </template>
      </div>

      <svg
        class="diagram-svg"
        viewBox="0 0 780 350"
        xmlns="http://www.w3.org/2000/svg"
        role="img"
        aria-label="Dirichlet root noise mixing over six illustrative moves including a zero-prior move"
      >
        <!-- Toy disclaimer -->
        <rect x="16" y="8" width="748" height="26" rx="6" fill="#fffbeb" stroke="#fcd34d" stroke-width="1.2" />
        <text x="390" y="25" text-anchor="middle" font-size="11.5" font-weight="600" fill="#92400e">
          Illustrative 6-move prior example (19×19 board has up to 361 legal moves) — not a measurement
        </text>

        <!-- Column headers -->
        <text x="24" y="58" font-size="12" font-weight="700" fill="#475569">Move</text>
        <text x="155" y="58" font-size="12" font-weight="700" fill="#475569">Prior P(s,a)</text>
        <text x="320" y="58" font-size="12" font-weight="700" fill="#475569">Noise η ~ Dir({{ alpha.toFixed(2) }})</text>
        <text x="485" y="58" font-size="12" font-weight="700" fill="#0f172a">
          Mixed P' = (1 − {{ epsilon.toFixed(2) }})P + {{ epsilon.toFixed(2) }}η
        </text>

        <!-- Move rows -->
        <g v-for="(move, idx) in MOVES" :key="move.label">
          <rect
            x="16"
            :y="68 + idx * 41"
            width="748"
            height="35"
            rx="6"
            :fill="idx === 5 ? '#f8fafc' : '#ffffff'"
            :stroke="idx === 5 && m6Mixed > 0.0005 ? '#16a34a' : '#e2e8f0'"
            :stroke-width="idx === 5 && m6Mixed > 0.0005 ? '1.5' : '1'"
          />

          <!-- Move label -->
          <text x="28" :y="90 + idx * 41" font-size="13" font-weight="700" fill="#0f172a">
            {{ move.label }}
          </text>
          <text x="64" :y="90 + idx * 41" font-size="11" fill="#64748b">
            {{ move.tag }}
          </text>

          <!-- Prior bar -->
          <rect x="155" :y="78 + idx * 41" width="95" height="14" rx="3" fill="#f1f5f9" />
          <rect
            v-if="move.prior > 0"
            x="155"
            :y="78 + idx * 41"
            :width="move.prior * 95"
            height="14"
            rx="3"
            fill="#64748b"
          />
          <text x="256" :y="89 + idx * 41" font-size="11.5" font-weight="600" fill="#334155">
            {{ move.prior.toFixed(3) }}
          </text>

          <!-- Dirichlet noise bar -->
          <rect x="320" :y="78 + idx * 41" width="95" height="14" rx="3" fill="#f1f5f9" />
          <rect
            v-if="noise[idx] > 0.001"
            x="320"
            :y="78 + idx * 41"
            :width="Math.max(2, noise[idx] * 95)"
            height="14"
            rx="3"
            fill="#d97706"
          />
          <text x="421" :y="89 + idx * 41" font-size="11.5" font-weight="600" fill="#92400e">
            {{ noise[idx].toFixed(3) }}
          </text>

          <!-- Mixed stacked bar: (1 - eps) * P + eps * eta -->
          <rect x="485" :y="78 + idx * 41" width="165" height="14" rx="3" fill="#f1f5f9" />
          <rect
            v-if="keptPart(idx) > 0.0005"
            x="485"
            :y="78 + idx * 41"
            :width="keptPart(idx) * 165"
            height="14"
            rx="3"
            fill="#2563eb"
          />
          <rect
            v-if="noisePart(idx) > 0.0005"
            :x="485 + keptPart(idx) * 165"
            :y="78 + idx * 41"
            :width="Math.max(2, noisePart(idx) * 165)"
            height="14"
            rx="3"
            :fill="idx === 5 ? '#16a34a' : '#f59e0b'"
          />
          <text
            x="658"
            :y="89 + idx * 41"
            font-size="12"
            font-weight="700"
            :fill="idx === 5 && m6Mixed > 0.0005 ? '#15803d' : '#0f172a'"
          >
            {{ mixed[idx].toFixed(3) }}
          </text>

          <!-- Child badge for M6 -->
          <text
            v-if="idx === 5"
            x="706"
            :y="89 + idx * 41"
            font-size="10.5"
            font-weight="700"
            :fill="m6Mixed > 0.0005 ? '#15803d' : '#94a3b8'"
          >
            {{ m6Mixed > 0.0005 ? 'child +' : 'no child' }}
          </text>
        </g>

        <!-- Legend -->
        <rect x="16" y="318" width="748" height="24" rx="5" fill="#ffffff" stroke="#e2e8f0" stroke-width="1" />
        <rect x="28" y="325" width="12" height="10" rx="2" fill="#2563eb" />
        <text x="45" y="334" font-size="11" fill="#334155">(1 − ε)P kept prior</text>
        <rect x="175" y="325" width="12" height="10" rx="2" fill="#f59e0b" />
        <text x="192" y="334" font-size="11" fill="#334155">εη added noise</text>
        <rect x="305" y="325" width="12" height="10" rx="2" fill="#16a34a" />
        <text x="322" y="334" font-size="11" fill="#15803d">
          zero-prior child created by _ensure_root_noise (baefedb)
        </text>
      </svg>

      <p class="source-note">
        Reflects <code>_safe_dirichlet_noise</code> and <code>_ensure_root_noise</code> in <code>search/engine.py</code>.
      </p>
    </div>
  </div>
</template>

<script setup>
import { computed, ref } from 'vue';

const MOVES = [
  { label: 'M1', tag: '(preferred)', prior: 0.54 },
  { label: 'M2', tag: '(reply)', prior: 0.24 },
  { label: 'M3', tag: '(alt)', prior: 0.12 },
  { label: 'M4', tag: '(edge)', prior: 0.06 },
  { label: 'M5', tag: '(rare)', prior: 0.04 },
  { label: 'M6', tag: '(P = 0)', prior: 0.0 },
];

// Standard Marsaglia-Tsang Gamma(k >= 1) sampler
function sampleGammaGte1(k) {
  const d = k - 1 / 3;
  const c = 1 / Math.sqrt(9 * d);
  for (;;) {
    const u1 = Math.max(1e-12, Math.random());
    const u2 = Math.random();
    const x = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    const v = Math.pow(1 + c * x, 3);
    if (v <= 0) continue;
    const u = Math.random();
    if (u < 1 - 0.0331 * Math.pow(x, 4)) return d * v;
    if (Math.log(Math.max(1e-12, u)) < 0.5 * x * x + d * (1 - v + Math.log(v))) {
      return d * v;
    }
  }
}

// Log-space Gamma(alpha) to avoid underflow when alpha = 0.03
function sampleLogGamma(a) {
  if (a >= 1) {
    return Math.log(Math.max(1e-30, sampleGammaGte1(a)));
  }
  const logG = Math.log(Math.max(1e-30, sampleGammaGte1(a + 1)));
  const logU = Math.log(Math.max(1e-12, Math.random()));
  return logG + logU / a;
}

function drawDirichlet(a, count) {
  const logs = [];
  for (let i = 0; i < count; i++) {
    logs.push(sampleLogGamma(a));
  }
  const maxLog = Math.max(...logs);
  const exps = logs.map((l) => Math.exp(l - maxLog));
  const sum = exps.reduce((acc, v) => acc + v, 0);
  return exps.map((v) => v / sum);
}

const epsilon = ref(0.25);
const alpha = ref(0.03);
const drawCount = ref(1);
// Deterministic initial sample representative of alpha = 0.03 spiking onto one move
const noise = ref([0.04, 0.0, 0.0, 0.78, 0.0, 0.18]);

const mixed = computed(() =>
  MOVES.map((m, idx) => (1 - epsilon.value) * m.prior + epsilon.value * noise.value[idx])
);

const m6Mixed = computed(() => mixed.value[5]);

function keptPart(idx) {
  return (1 - epsilon.value) * MOVES[idx].prior;
}

function noisePart(idx) {
  return epsilon.value * noise.value[idx];
}

function isClose(a, b) {
  return Math.abs(a - b) < 0.005;
}

function resampleNoise() {
  noise.value = drawDirichlet(alpha.value, MOVES.length);
  drawCount.value += 1;
}

function resampleWithAlpha(newAlpha) {
  alpha.value = newAlpha;
  resampleNoise();
}

function spikeOnZeroPrior() {
  const rest = drawDirichlet(Math.max(0.05, alpha.value), MOVES.length - 1);
  const spike = 0.72;
  noise.value = [...rest.map((v) => v * (1 - spike)), spike];
  drawCount.value += 1;
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
  gap: 10px;
  margin-bottom: 10px;
}

.slider-group {
  display: flex;
  flex-wrap: wrap;
  justify-content: center;
  gap: 12px 20px;
}

.slider-item {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: center;
  gap: 8px;
  font-size: 13px;
  color: #334155;
}

.range-input {
  width: 130px;
  max-width: 100%;
  accent-color: #2563eb;
  cursor: pointer;
}

.preset-btns,
.action-row {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: center;
  gap: 8px;
}

.preset-btn,
.action-btn {
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

.preset-btn:hover,
.action-btn:hover {
  background: #f1f5f9;
  border-color: #94a3b8;
}

.preset-btn.active {
  background: #eff6ff;
  border-color: #2563eb;
  color: #1e40af;
}

.action-btn.primary {
  background: #2563eb;
  border-color: #2563eb;
  color: #ffffff;
}

.action-btn.primary:hover {
  background: #1d4ed8;
}

.draw-badge {
  font-size: 12px;
  font-weight: 600;
  color: #475569;
  background: #ffffff;
  border: 1px solid #e2e8f0;
  border-radius: 6px;
  padding: 4px 8px;
}

.status-bar {
  background: #ffffff;
  border: 1.5px solid #e2e8f0;
  border-radius: 8px;
  padding: 9px 13px;
  margin-bottom: 10px;
  font-size: 12.5px;
  line-height: 1.45;
  color: #334155;
}

.status-bar.is-rescued {
  background: #f0fdf4;
  border-color: #86efac;
}

.status-bar.is-zero {
  background: #fffbeb;
  border-color: #fcd34d;
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
}
</style>
