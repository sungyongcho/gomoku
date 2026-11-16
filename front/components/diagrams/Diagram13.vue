<template>
  <div class="diagram-wrapper">
    <div class="container">
      <h1>Affine 8-Bit Quantization: Float32 ↔ 256 Integer Levels</h1>

      <div class="controls-panel">
        <div class="mode-row">
          <button
            type="button"
            class="mode-btn"
            :class="{ active: quantType === 'uint8' }"
            @click="quantType = 'uint8'"
          >
            QUInt8 [0, 255] (activations)
          </button>
          <button
            type="button"
            class="mode-btn"
            :class="{ active: quantType === 'int8' }"
            @click="quantType = 'int8'"
          >
            QInt8 [−128, 127] (weights)
          </button>
        </div>

        <div class="sliders-row">
          <label class="slider-item">
            <span><code>x_min</code>: <strong>{{ xMin.toFixed(1) }}</strong></span>
            <input v-model.number="xMin" type="range" min="-3.0" max="0.0" step="0.1" class="range-input" />
          </label>
          <label class="slider-item">
            <span><code>x_max</code>: <strong>{{ xMax.toFixed(1) }}</strong></span>
            <input v-model.number="xMax" type="range" min="0.2" max="3.0" step="0.1" class="range-input" />
          </label>
          <label class="slider-item">
            <span>Input <code>x</code>: <strong>{{ fmtSigned(xVal, 2) }}</strong></span>
            <input v-model.number="xVal" type="range" min="-3.5" max="3.5" step="0.01" class="range-input" />
          </label>
          <div class="preset-btns">
            <button type="button" class="preset-btn" @click="xVal = 0.0">x = 0.00</button>
            <button type="button" class="preset-btn" @click="xVal = 0.73">x = 0.73</button>
            <button type="button" class="preset-btn" @click="xVal = 2.10">x = 2.10 (clip)</button>
          </div>
        </div>
      </div>

      <svg
        class="diagram-svg"
        viewBox="0 0 780 370"
        xmlns="http://www.w3.org/2000/svg"
        role="img"
        aria-label="Illustrative affine 8-bit quantization mapping a float32 range onto 256 integer levels"
      >
        <!-- Illustrative disclaimer -->
        <rect x="20" y="8" width="740" height="24" rx="6" fill="#fffbeb" stroke="#fcd34d" stroke-width="1.2" />
        <text x="390" y="24" text-anchor="middle" font-size="11.5" font-weight="600" fill="#92400e">
          Illustrative example (formula and QUInt8 / QInt8 ranges from the doc; arbitrary bounds and sample x)
        </text>

        <!-- Section 1: Float32 axis and 256-level integer range -->
        <text x="24" y="50" font-size="12.5" font-weight="700" fill="#0f172a">
          Float32 Input Range (−3.5 to +3.5) with Calibrated Window [{{ xMin.toFixed(1) }}, {{ xMax.toFixed(1) }}]
        </text>

        <!-- Calibrated mapping trapezoid -->
        <polygon
          :points="`${realToX(xMin)},74 ${realToX(xMax)},74 720,126 60,126`"
          fill="#dbeafe"
          opacity="0.45"
        />
        <line :x1="realToX(xMin)" y1="74" :x2="60" y2="126" stroke="#2563eb" stroke-width="1.5" stroke-dasharray="4 3" />
        <line :x1="realToX(xMax)" y1="74" :x2="720" y2="126" stroke="#2563eb" stroke-width="1.5" stroke-dasharray="4 3" />

        <!-- Real axis base -->
        <line x1="60" y1="74" x2="720" y2="74" stroke="#94a3b8" stroke-width="2" />
        <!-- Calibrated span highlight -->
        <line :x1="realToX(xMin)" y1="74" :x2="realToX(xMax)" y2="74" stroke="#2563eb" stroke-width="5" stroke-linecap="round" />

        <!-- Real axis ticks -->
        <g v-for="t in [-3, -2, -1, 0, 1, 2, 3]" :key="'rt-' + t">
          <line :x1="realToX(t)" y1="69" :x2="realToX(t)" y2="79" stroke="#64748b" stroke-width="1.3" />
          <text :x="realToX(t)" y="93" text-anchor="middle" font-size="11" fill="#475569">{{ t }}</text>
        </g>

        <!-- Input x marker on real axis -->
        <line :x1="realToX(xVal)" y1="68" :x2="realToX(xVal)" y2="82" stroke="#1d4ed8" stroke-width="2" />
        <circle :cx="realToX(xVal)" cy="74" r="5" fill="#2563eb" stroke="#ffffff" stroke-width="1.5" />
        <text :x="clampLabelX(realToX(xVal))" y="62" text-anchor="middle" font-size="11.5" font-weight="700" fill="#1d4ed8">
          x = {{ fmtSigned(xVal, 2) }}
        </text>

        <!-- 256-level integer bar -->
        <rect x="60" y="126" width="660" height="22" rx="4" fill="#f1f5f9" stroke="#475569" stroke-width="1.5" />
        <text x="60" y="164" text-anchor="start" font-size="11.5" font-weight="600" fill="#334155">
          q_min = {{ qMin }}
        </text>
        <text :x="zLabelX" y="181" :text-anchor="zLabelAnchor" font-size="11.5" font-weight="700" fill="#7c3aed">
          z = {{ zeroPoint }} (0.0)
        </text>
        <text x="720" y="164" text-anchor="end" font-size="11.5" font-weight="600" fill="#334155">
          q_max = {{ qMax }}
        </text>

        <!-- Zero-point tick on int bar -->
        <line :x1="intToX(zeroPoint)" y1="124" :x2="intToX(zeroPoint)" y2="150" stroke="#7c3aed" stroke-width="2" />
        <!-- Quantized q marker on int bar -->
        <circle :cx="intToX(qVal)" cy="137" r="5.5" :fill="isClipped ? '#dc2626' : '#16a34a'" stroke="#ffffff" stroke-width="1.5" />
        <text
          :x="clampLabelX(intToX(qVal))"
          y="120"
          text-anchor="middle"
          font-size="11.5"
          font-weight="700"
          :fill="isClipped ? '#dc2626' : '#15803d'"
        >
          q = {{ qVal }}{{ isClipped ? ' (clipped)' : '' }}
        </text>

        <!-- Section 2 left: Zoomed integer grid around q -->
        <rect x="20" y="182" width="468" height="176" rx="8" fill="#ffffff" stroke="#cbd5e1" stroke-width="1.5" />
        <text x="34" y="204" font-size="13" font-weight="700" fill="#0f172a">
          Zoomed Grid Around q = {{ qVal }} (step s = {{ scale.toFixed(4) }})
        </text>

        <!-- Zoom axis line -->
        <line x1="56" y1="268" x2="452" y2="268" stroke="#94a3b8" stroke-width="2" />

        <!-- 7 consecutive integer ticks -->
        <g v-for="(lvl, i) in zoomLevels" :key="'zl-' + lvl">
          <line
            :x1="zoomIndexToX(i)"
            y1="254"
            :x2="zoomIndexToX(i)"
            y2="282"
            :stroke="lvl === qVal ? (isClipped ? '#dc2626' : '#16a34a') : '#64748b'"
            :stroke-width="lvl === qVal ? '2.5' : '1.4'"
          />
          <text
            :x="zoomIndexToX(i)"
            y="246"
            text-anchor="middle"
            font-size="12"
            :font-weight="lvl === qVal ? '700' : '500'"
            :fill="lvl === qVal ? (isClipped ? '#dc2626' : '#15803d') : '#334155'"
          >
            {{ lvl }}
          </text>
          <text :x="zoomIndexToX(i)" y="299" text-anchor="middle" font-size="10.5" fill="#64748b">
            {{ dequant(lvl).toFixed(3) }}
          </text>
        </g>

        <!-- Exact x marker & error bracket inside zoom window -->
        <template v-if="!isClipped">
          <line
            :x1="zoomRealToX(xVal)"
            y1="258"
            :x2="zoomRealToX(xVal)"
            y2="278"
            stroke="#2563eb"
            stroke-width="2.5"
          />
          <circle :cx="zoomRealToX(xVal)" cy="268" r="4.5" fill="#2563eb" />
          <line
            :x1="zoomRealToX(xVal)"
            y1="318"
            :x2="zoomRealToX(xHat)"
            y2="318"
            stroke="#d97706"
            stroke-width="3"
          />
          <text x="254" y="340" text-anchor="middle" font-size="12" font-weight="600" fill="#92400e">
            Rounding error x̂ − x = {{ fmtSigned(error, 4) }} (max step ±s/2 = ±{{ (scale / 2).toFixed(4) }})
          </text>
        </template>
        <template v-else>
          <text x="254" y="336" text-anchor="middle" font-size="12" font-weight="700" fill="#dc2626">
            x = {{ fmtSigned(xVal, 2) }} is outside [{{ xMin.toFixed(1) }}, {{ xMax.toFixed(1) }}] → saturated at q = {{ qVal }} (error {{ fmtSigned(error, 3) }})
          </text>
        </template>

        <!-- Section 2 right: Equation readout box -->
        <rect x="502" y="182" width="258" height="176" rx="8" fill="#ffffff" stroke="#cbd5e1" stroke-width="1.5" />
        <text x="516" y="205" font-size="13" font-weight="700" fill="#0f172a">
          Quantization Values
        </text>
        <text x="516" y="230" font-size="12.5" fill="#334155">
          s = (x_max − x_min) / 255 = <tspan font-weight="700">{{ scale.toFixed(5) }}</tspan>
        </text>
        <text x="516" y="254" font-size="12.5" fill="#7c3aed">
          z = round(q_min − x_min/s) = <tspan font-weight="700">{{ zeroPoint }}</tspan>
        </text>
        <text x="516" y="278" font-size="12.5" fill="#1d4ed8">
          q = clip(round(x/s) + z) = <tspan font-weight="700">{{ qVal }}</tspan>
        </text>
        <text x="516" y="302" font-size="12.5" :fill="isClipped ? '#dc2626' : '#15803d'">
          x̂ = s · (q − z) = <tspan font-weight="700">{{ fmtSigned(xHat, 4) }}</tspan>
        </text>
        <text x="516" y="328" font-size="12.5" font-weight="700" :fill="isClipped ? '#dc2626' : '#92400e'">
          Error (x̂ − x) = {{ fmtSigned(error, 4) }}
        </text>
        <text x="516" y="346" font-size="11" fill="#64748b">
          Note: q = z maps exactly to 0.0000
        </text>
      </svg>
    </div>
  </div>
</template>

<script setup>
import { computed, ref } from 'vue';

const quantType = ref('uint8');
const xMin = ref(-1.0);
const xMax = ref(1.5);
const xVal = ref(0.73);

const qMin = computed(() => (quantType.value === 'uint8' ? 0 : -128));
const qMax = computed(() => (quantType.value === 'uint8' ? 255 : 127));

const scale = computed(() => (xMax.value - xMin.value) / 255);

const clampInt = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

const zeroPoint = computed(() =>
  clampInt(Math.round(qMin.value - xMin.value / scale.value), qMin.value, qMax.value)
);

const rawQ = computed(() => Math.round(xVal.value / scale.value) + zeroPoint.value);
const qVal = computed(() => clampInt(rawQ.value, qMin.value, qMax.value));
const isClipped = computed(() => rawQ.value < qMin.value || rawQ.value > qMax.value);

const dequant = (q) => scale.value * (q - zeroPoint.value);
const xHat = computed(() => dequant(qVal.value));
const error = computed(() => xHat.value - xVal.value);

const zoomStart = computed(() => clampInt(qVal.value - 3, qMin.value, qMax.value - 6));
const zoomLevels = computed(() => Array.from({ length: 7 }, (_, i) => zoomStart.value + i));

function realToX(val) {
  return 60 + ((val + 3.5) / 7.0) * 660;
}

function intToX(q) {
  return 60 + ((q - qMin.value) / 255) * 660;
}

// The zero-point label sits on its own row and turns inward near either end of the bar.
const zLabelX = computed(() => intToX(zeroPoint.value));
const zLabelAnchor = computed(() => {
  if (zLabelX.value < 110) return 'start';
  if (zLabelX.value > 670) return 'end';
  return 'middle';
});

function clampLabelX(x) {
  return Math.min(715, Math.max(65, x));
}

function zoomIndexToX(i) {
  return 56 + (i / 6) * 396;
}

function zoomRealToX(val) {
  const startX = dequant(zoomStart.value);
  const endX = dequant(zoomStart.value + 6);
  const frac = (val - startX) / (endX - startX);
  return 56 + Math.min(1, Math.max(0, frac)) * 396;
}

function fmtSigned(val, decimals) {
  const fixed = val.toFixed(decimals);
  return val >= 0 ? `+${fixed}` : fixed;
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

.mode-row,
.sliders-row {
  display: flex;
  flex-wrap: wrap;
  justify-content: center;
  align-items: center;
  gap: 10px;
}

.mode-btn,
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

.mode-btn:hover,
.preset-btn:hover {
  background: #f1f5f9;
  border-color: #94a3b8;
}

.mode-btn.active {
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
  width: 125px;
  max-width: 100%;
  accent-color: #2563eb;
  cursor: pointer;
}

.preset-btns {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
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
