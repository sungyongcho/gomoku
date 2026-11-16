<template>
  <div class="diagram-wrapper">
    <div class="container">
      <h1>Unused Weight Under Adam+L2 vs. AdamW</h1>

      <div class="controls-panel">
        <div class="preset-row">
          <button
            v-for="preset in presets"
            :key="preset.id"
            type="button"
            class="preset-btn"
            :class="{ active: activePresetId === preset.id }"
            @click="applyPreset(preset)"
          >
            {{ preset.label }}
          </button>
        </div>

        <div class="sliders-row">
          <label class="slider-item">
            <span>Initial <code>θ₀</code>: <strong>{{ theta0.toFixed(2) }}</strong></span>
            <input v-model.number="theta0" type="range" min="0.10" max="1.00" step="0.05" class="range-input" />
          </label>
          <label class="slider-item">
            <span>Step <code>t</code>: <strong>{{ currentStep }}</strong> / {{ maxSteps }}</span>
            <input v-model.number="currentStep" type="range" min="0" :max="maxSteps" step="5" class="range-input" />
          </label>
        </div>
      </div>

      <svg
        class="diagram-svg"
        viewBox="0 0 780 392"
        xmlns="http://www.w3.org/2000/svg"
        role="img"
        aria-label="Illustrative comparison of an unused scalar weight under Adam with L2 regularization versus AdamW"
      >
        <!-- Illustrative banner -->
        <rect x="20" y="8" width="740" height="24" rx="6" fill="#fffbeb" stroke="#fcd34d" stroke-width="1.2" />
        <text x="390" y="24" text-anchor="middle" font-size="11.5" font-weight="600" fill="#92400e">
          Illustrative toy simulation (scalar weight θ with zero loss gradient g_t = 0; PyTorch Adam vs. AdamW)
        </text>

        <!-- Left chart card -->
        <rect x="20" y="42" width="466" height="220" rx="8" fill="#ffffff" stroke="#cbd5e1" stroke-width="1.4" />
        <text x="34" y="61" font-size="12.5" font-weight="700" fill="#0f172a">
          Magnitude |θ_t| over {{ maxSteps }} Steps (η = {{ lr }}, λ = {{ wd }})
        </text>

        <!-- Legend above plot area -->
        <line x1="36" y1="77" x2="58" y2="77" stroke="#2563eb" stroke-width="2.6" />
        <text x="64" y="81" font-size="11" font-weight="600" fill="#1e40af">AdamW (decoupled)</text>
        <line x1="194" y1="77" x2="216" y2="77" stroke="#dc2626" stroke-width="2.6" />
        <text x="222" y="81" font-size="11" font-weight="600" fill="#b91c1c">Adam + L2</text>

        <!-- Horizontal grid lines -->
        <g v-for="tick in [0, 0.25, 0.5, 0.75, 1.0]" :key="'y-' + tick">
          <line
            x1="72"
            :y1="valToY(tick)"
            x2="466"
            :y2="valToY(tick)"
            stroke="#e2e8f0"
            stroke-width="1"
          />
          <text x="66" :y="valToY(tick) + 4" text-anchor="end" font-size="10.5" fill="#64748b">
            {{ tick.toFixed(2) }}
          </text>
        </g>

        <!-- Vertical grid / X ticks -->
        <g v-for="stepTick in [0, 150, 300, 450, 600]" :key="'x-' + stepTick">
          <line
            :x1="stepToX(stepTick)"
            y1="94"
            :x2="stepToX(stepTick)"
            y2="232"
            stroke="#f1f5f9"
            stroke-width="1"
          />
          <text :x="stepToX(stepTick)" y="248" text-anchor="middle" font-size="10.5" fill="#64748b">
            {{ stepTick }}
          </text>
        </g>

        <!-- Axes -->
        <line x1="72" y1="94" x2="72" y2="232" stroke="#94a3b8" stroke-width="1.4" />
        <line x1="72" y1="232" x2="466" y2="232" stroke="#94a3b8" stroke-width="1.4" />
        <text x="269" y="259" text-anchor="middle" font-size="10.5" font-weight="600" fill="#475569">
          Optimizer step t
        </text>

        <!-- Curves -->
        <polyline
          :points="adamWPoints"
          fill="none"
          stroke="#2563eb"
          stroke-width="2.6"
          stroke-linejoin="round"
        />
        <polyline
          :points="adamL2Points"
          fill="none"
          stroke="#dc2626"
          stroke-width="2.6"
          stroke-linejoin="round"
        />

        <!-- Selected step marker -->
        <line
          :x1="stepToX(currentStep)"
          y1="94"
          :x2="stepToX(currentStep)"
          y2="232"
          stroke="#475569"
          stroke-width="1.3"
          stroke-dasharray="4 3"
        />
        <circle
          :cx="stepToX(currentStep)"
          :cy="valToY(Math.abs(adamWSeries[currentStep]))"
          r="4.5"
          fill="#2563eb"
          stroke="#ffffff"
          stroke-width="1.5"
        />
        <circle
          :cx="stepToX(currentStep)"
          :cy="valToY(Math.abs(adamL2Series[currentStep]))"
          r="4.5"
          fill="#dc2626"
          stroke="#ffffff"
          stroke-width="1.5"
        />

        <!-- Right readout panel -->
        <rect x="498" y="42" width="262" height="220" rx="8" fill="#ffffff" stroke="#cbd5e1" stroke-width="1.4" />
        <text x="512" y="62" font-size="12.5" font-weight="700" fill="#0f172a">
          State at Step t = {{ currentStep }}
        </text>

        <!-- Adam + L2 readout -->
        <rect x="510" y="74" width="238" height="74" rx="6" fill="#fef2f2" stroke="#fca5a5" stroke-width="1.2" />
        <text x="520" y="92" font-size="11.5" font-weight="700" fill="#991b1b">
          Adam + L2 (optimizer: "adam")
        </text>
        <text x="520" y="110" font-size="11.5" fill="#7f1d1d">
          θ_t = {{ formatVal(adamL2Series[currentStep]) }} ({{ pctOfInit(adamL2Series[currentStep]) }}%)
        </text>
        <text x="520" y="126" font-size="10.5" fill="#991b1b">
          Last step |Δθ| = {{ formatStep(adamL2LastDelta) }} (η = {{ lr }})
        </text>
        <text x="520" y="141" font-size="10" fill="#b91c1c">
          Also decays Conv bias &amp; BatchNorm γ, β
        </text>

        <!-- AdamW readout -->
        <rect x="510" y="158" width="238" height="92" rx="6" fill="#eff6ff" stroke="#93c5fd" stroke-width="1.2" />
        <text x="520" y="176" font-size="11.5" font-weight="700" fill="#1e40af">
          AdamW (optimizer: "adamw")
        </text>
        <text x="520" y="194" font-size="11.5" fill="#1e3a8a">
          Weight θ_t = {{ formatVal(adamWSeries[currentStep]) }} ({{ pctOfInit(adamWSeries[currentStep]) }}%)
        </text>
        <text x="520" y="210" font-size="10.5" fill="#1e40af">
          Last step |Δθ| = {{ formatStep(adamWLastDelta) }} (= η·λ·|θ|)
        </text>
        <text x="520" y="227" font-size="10.5" font-weight="600" fill="#047857">
          BN γ (1D, wd = 0): 1.0000 (100% kept)
        </text>
        <text x="520" y="242" font-size="10" fill="#1d4ed8">
          Matches build_optimizer param groups
        </text>

        <!-- Bottom mechanism cards -->
        <rect x="20" y="274" width="362" height="106" rx="8" fill="#fef2f2" stroke="#fca5a5" stroke-width="1.4" />
        <text x="34" y="295" font-size="12" font-weight="700" fill="#991b1b">
          Why Adam + L2 Kills Unused Channels
        </text>
        <text x="34" y="314" font-size="11" fill="#7f1d1d">
          1. L2 folds into gradient: g̃_t = g_t + λθ_t = λθ_t
        </text>
        <text x="34" y="332" font-size="11" fill="#7f1d1d">
          2. Adaptive ratio cancels λ: m̂_t / (√v̂_t + ε) ≈ sign(θ_t)
        </text>
        <text x="34" y="350" font-size="11" fill="#7f1d1d">
          3. Weight marches toward 0 at full lr η until |λθ_t| ~ ε,
        </text>
        <text x="48" y="366" font-size="11" fill="#7f1d1d">
          leaving dead filters and float32 subnormal weights.
        </text>

        <rect x="398" y="274" width="362" height="106" rx="8" fill="#eff6ff" stroke="#93c5fd" stroke-width="1.4" />
        <text x="412" y="295" font-size="12" font-weight="700" fill="#1e40af">
          Why AdamW Keeps Channels Alive
        </text>
        <text x="412" y="314" font-size="11" fill="#1e3a8a">
          1. Adaptive moments use only loss gradient: g_t = 0 ⇒ m̂_t = 0
        </text>
        <text x="412" y="332" font-size="11" fill="#1e3a8a">
          2. Decay is decoupled: θ_{t+1} = θ_t − ηλθ_t = θ_t(1 − ηλ)
        </text>
        <text x="412" y="350" font-size="11" fill="#1e3a8a">
          3. Decay scales with η·λ·θ_t; 1D params (BatchNorm γ, β,
        </text>
        <text x="426" y="366" font-size="11" fill="#1e3a8a">
          biases) get weight_decay = 0.0 in build_optimizer.
        </text>
      </svg>
    </div>
  </div>
</template>

<script setup>
import { computed, ref } from 'vue';

const presets = [
  { id: 'v5', label: 'v5 / v6 (η=0.001, λ=0.0001)', lr: 0.001, wd: 0.0001 },
  { id: 'v6b', label: 'v6b (η=0.0003, λ=0.01)', lr: 0.0003, wd: 0.01 },
  { id: 'test', label: 'Unit test (η=0.01, λ=0.1)', lr: 0.01, wd: 0.1 },
];

const maxSteps = 600;
const activePresetId = ref('v5');
const lr = ref(0.001);
const wd = ref(0.0001);
const theta0 = ref(0.5);
const currentStep = ref(300);

function applyPreset(preset) {
  activePresetId.value = preset.id;
  lr.value = preset.lr;
  wd.value = preset.wd;
}

const sim = computed(() => {
  const beta1 = 0.9;
  const beta2 = 0.999;
  const eps = 1e-8;
  const eta = lr.value;
  const lam = wd.value;

  const adamL2 = [theta0.value];
  const adamW = [theta0.value];

  let th1 = theta0.value;
  let m1 = 0;
  let v1 = 0;

  let th2 = theta0.value;

  for (let t = 1; t <= maxSteps; t++) {
    // Adam + L2 with g_t = 0
    const gTilde = lam * th1;
    m1 = beta1 * m1 + (1 - beta1) * gTilde;
    v1 = beta2 * v1 + (1 - beta2) * gTilde * gTilde;
    const mHat = m1 / (1 - Math.pow(beta1, t));
    const vHat = v1 / (1 - Math.pow(beta2, t));
    th1 = th1 - eta * (mHat / (Math.sqrt(vHat) + eps));
    adamL2.push(th1);

    // AdamW with g_t = 0 -> adaptive step is 0, only decoupled decay applies
    th2 = th2 * (1 - eta * lam);
    adamW.push(th2);
  }

  return { adamL2, adamW };
});

const adamL2Series = computed(() => sim.value.adamL2);
const adamWSeries = computed(() => sim.value.adamW);

const adamL2LastDelta = computed(() => {
  const t = currentStep.value;
  if (t === 0) return 0;
  return Math.abs(adamL2Series.value[t] - adamL2Series.value[t - 1]);
});

const adamWLastDelta = computed(() => {
  const t = currentStep.value;
  if (t === 0) return 0;
  return Math.abs(adamWSeries.value[t] - adamWSeries.value[t - 1]);
});

function stepToX(t) {
  return 72 + (t / maxSteps) * (466 - 72);
}

function valToY(v) {
  const clamped = Math.min(1.0, Math.max(0.0, v));
  return 232 - clamped * (232 - 94);
}

const adamL2Points = computed(() =>
  adamL2Series.value
    .map((v, i) => `${stepToX(i).toFixed(1)},${valToY(Math.abs(v)).toFixed(1)}`)
    .join(' ')
);

const adamWPoints = computed(() =>
  adamWSeries.value
    .map((v, i) => `${stepToX(i).toFixed(1)},${valToY(Math.abs(v)).toFixed(1)}`)
    .join(' ')
);

function formatVal(v) {
  const av = Math.abs(v);
  if (av === 0) return '0.0000';
  if (av < 1e-3) return v.toExponential(2);
  return v.toFixed(4);
}

function formatStep(d) {
  if (d === 0) return '0';
  if (d < 1e-4) return d.toExponential(2);
  return d.toFixed(5);
}

function pctOfInit(v) {
  return ((Math.abs(v) / theta0.value) * 100).toFixed(1);
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

.preset-row,
.sliders-row {
  display: flex;
  flex-wrap: wrap;
  justify-content: center;
  align-items: center;
  gap: 8px 12px;
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

.slider-item {
  display: flex;
  align-items: center;
  gap: 7px;
  font-size: 12.5px;
  color: #334155;
}

.range-input {
  width: 130px;
  max-width: 100%;
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
