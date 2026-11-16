<template>
  <div class="diagram-wrapper">
    <div class="container">
      <h1>Time-Limited Search Budget (Two Neoverse-N1 Cores)</h1>

      <div class="controls-group">
        <div class="row-buttons" role="group" aria-label="Measured configurations">
          <button
            v-for="row in ROWS"
            :key="row.id"
            type="button"
            class="cfg-btn"
            :class="{ active: selectedId === row.id }"
            @click="selectedId = row.id"
          >
            {{ row.shortLabel }}
          </button>
        </div>

        <div class="slider-row">
          <label for="budget-slider" class="slider-label">
            Time budget <code>T</code>: <strong>{{ timeBudget.toFixed(1) }} s</strong>
          </label>
          <input
            id="budget-slider"
            v-model.number="timeBudget"
            type="range"
            min="0"
            max="3"
            step="0.1"
            class="range-input"
          />
          <div class="preset-btns">
            <button type="button" class="preset-btn" :class="{ active: isExact(1.0) }" @click="timeBudget = 1.0">1.0 s</button>
            <button type="button" class="preset-btn" :class="{ active: isExact(2.5) }" @click="timeBudget = 2.5">2.5 s (limit)</button>
            <button type="button" class="preset-btn" :class="{ active: isExact(3.0) }" @click="timeBudget = 3.0">3.0 s (target)</button>
          </div>
        </div>
      </div>

      <div class="status-bar" :class="{ 'is-measured': isMeasuredBudget, 'is-over': timeBudget > 2.5 }">
        <div class="status-main">
          <span class="badge" :class="isMeasuredBudget ? 'badge-green' : 'badge-amber'">
            {{ isMeasuredBudget ? 'Measured at 2.5 s' : 'Linear estimate (not measured)' }}
          </span>
          <span class="status-text">
            <strong>{{ activeRow.label }}</strong>:
            <template v-if="isMeasuredBudget">
              <strong>{{ activeRow.measured25 }} searches</strong> at 2.5 s ({{ activeRate.toFixed(1) }} / s)
            </template>
            <template v-else>
              <strong>~{{ estimateFor(activeRow) }} searches</strong> at {{ timeBudget.toFixed(1) }} s
              (linear from measured {{ activeRow.measured25 }} at 2.5 s = {{ activeRate.toFixed(1) }} / s)
            </template>
          </span>
        </div>
        <div class="status-margin">
          <template v-if="timeBudget <= 2.5">
            Margin to 3.0 s target: <strong>{{ (3.0 - timeBudget).toFixed(1) }} s</strong>
            <span v-if="isExact(2.5)"> (0.5 s for final leaf eval + serialization)</span>
          </template>
          <template v-else>
            Enters the 0.5 s safety margin inside the 3.0 s move target (only {{ (3.0 - timeBudget).toFixed(1) }} s left for final eval + serialization).
          </template>
        </div>
      </div>

      <svg
        class="diagram-svg"
        viewBox="0 0 780 375"
        xmlns="http://www.w3.org/2000/svg"
        role="img"
        aria-label="Time budget timeline and search counts across four measured configurations"
      >
        <!-- Panel 1: Time axis -->
        <text x="24" y="24" font-size="14" font-weight="700" fill="#0f172a">
          Clock Window vs. 3.0 s Move Target
        </text>

        <!-- 2.5 s and 3.0 s guide labels above track -->
        <text :x="timeToX(2.5) - 8" y="46" text-anchor="end" font-size="11.5" font-weight="700" fill="#15803d">
          2.5 s serving limit
        </text>
        <text :x="timeToX(3.0)" y="46" text-anchor="end" font-size="11.5" font-weight="700" fill="#dc2626">
          3.0 s target
        </text>

        <!-- Track background -->
        <rect x="44" y="54" width="690" height="34" rx="6" fill="#f1f5f9" stroke="#cbd5e1" stroke-width="1.5" />

        <!-- 0.5 s margin shading (2.5 s to 3.0 s) -->
        <rect :x="timeToX(2.5)" y="54" :width="timeToX(3.0) - timeToX(2.5)" height="34" rx="4" fill="#fef3c7" opacity="0.9" />
        <text
          :x="(timeToX(2.5) + timeToX(3.0)) / 2"
          y="75"
          text-anchor="middle"
          font-size="11.5"
          font-weight="600"
          fill="#92400e"
        >
          0.5 s margin
        </text>

        <!-- Active budget bar up to min(2.5, timeBudget) -->
        <rect
          v-if="timeBudget > 0"
          x="44"
          y="54"
          :width="Math.max(0, timeToX(Math.min(timeBudget, 2.5)) - 44)"
          height="34"
          rx="5"
          :fill="isMeasuredBudget ? '#16a34a' : '#2563eb'"
          opacity="0.88"
        />

        <!-- Overflow into 0.5 s margin if timeBudget > 2.5 -->
        <rect
          v-if="timeBudget > 2.5"
          :x="timeToX(2.5)"
          y="54"
          :width="Math.max(0, timeToX(timeBudget) - timeToX(2.5))"
          height="34"
          rx="2"
          fill="#d97706"
          opacity="0.85"
        />

        <!-- Label inside active bar when wide enough -->
        <text
          v-if="timeBudget >= 0.7"
          :x="44 + (timeToX(Math.min(timeBudget, 2.5)) - 44) / 2"
          y="75"
          text-anchor="middle"
          font-size="12.5"
          font-weight="700"
          fill="#ffffff"
        >
          {{ isMeasuredBudget ? `${activeRow.measured25} searches (measured)` : `~${estimateFor(activeRow)} est.` }}
        </text>

        <!-- Time ticks -->
        <g v-for="t in TIME_TICKS" :key="t">
          <line :x1="timeToX(t)" y1="88" :x2="timeToX(t)" y2="96" stroke="#64748b" stroke-width="1.5" />
          <text :x="timeToX(t)" y="112" text-anchor="middle" font-size="12" fill="#475569">
            {{ t.toFixed(1) }} s
          </text>
        </g>

        <!-- 2.5 s serving limit guide -->
        <line
          :x1="timeToX(2.5)"
          y1="48"
          :x2="timeToX(2.5)"
          y2="94"
          stroke="#15803d"
          stroke-width="2"
          stroke-dasharray="4 3"
        />

        <!-- 3.0 s move target guide -->
        <line
          :x1="timeToX(3.0)"
          y1="48"
          :x2="timeToX(3.0)"
          y2="94"
          stroke="#dc2626"
          stroke-width="2"
          stroke-dasharray="4 3"
        />

        <!-- Current slider marker -->
        <line
          :x1="timeToX(timeBudget)"
          y1="49"
          :x2="timeToX(timeBudget)"
          y2="93"
          stroke="#0f172a"
          stroke-width="2.5"
        />
        <circle :cx="timeToX(timeBudget)" cy="49" r="4.5" fill="#0f172a" />

        <!-- Divider -->
        <line x1="24" y1="132" x2="756" y2="132" stroke="#e2e8f0" stroke-width="1.5" />

        <!-- Panel 2: All 4 measured configurations -->
        <text x="24" y="158" font-size="14" font-weight="700" fill="#0f172a">
          {{ isMeasuredBudget ? 'Measured Searches at 2.5 s' : `Linear Estimate at ${timeBudget.toFixed(1)} s (from 2.5 s rate)` }}
        </text>
        <text x="756" y="158" text-anchor="end" font-size="12" fill="#64748b">
          Click a row to select · dashed tick = measured 2.5 s count
        </text>

        <g
          v-for="(row, idx) in ROWS"
          :key="row.id"
          class="bar-row"
          @click="selectedId = row.id"
        >
          <!-- Row background highlight when selected -->
          <rect
            x="16"
            :y="174 + idx * 46"
            width="748"
            height="40"
            rx="6"
            :fill="selectedId === row.id ? '#eff6ff' : 'transparent'"
            :stroke="selectedId === row.id ? '#93c5fd' : 'transparent'"
            stroke-width="1.5"
          />

          <!-- Config label -->
          <text
            x="26"
            :y="199 + idx * 46"
            font-size="13"
            :font-weight="selectedId === row.id ? '700' : '600'"
            :fill="selectedId === row.id ? '#1e3a8a' : '#334155'"
          >
            {{ row.shortLabel }}
          </text>

          <!-- Bar track -->
          <rect
            x="232"
            :y="182 + idx * 46"
            width="430"
            height="24"
            rx="4"
            fill="#f1f5f9"
            stroke="#e2e8f0"
            stroke-width="1"
          />

          <!-- Current value bar -->
          <rect
            v-if="estimateFor(row) > 0"
            x="232"
            :y="182 + idx * 46"
            :width="countToW(estimateFor(row))"
            height="24"
            rx="4"
            :fill="barColor(row.id)"
          />

          <!-- Reference tick at measured 2.5 s count -->
          <line
            :x1="232 + countToW(row.measured25)"
            :y1="178 + idx * 46"
            :x2="232 + countToW(row.measured25)"
            :y2="210 + idx * 46"
            stroke="#0f172a"
            stroke-width="2"
            stroke-dasharray="3 2"
            opacity="0.7"
          />

          <!-- Count text -->
          <text
            x="666"
            :y="199 + idx * 46"
            :font-size="isMeasuredBudget ? 13 : 12"
            :font-weight="selectedId === row.id ? '700' : '600'"
            :fill="selectedId === row.id ? '#0f172a' : '#475569'"
          >
            <template v-if="isMeasuredBudget">{{ row.measured25 }}</template>
            <template v-else>~{{ estimateFor(row) }} (2.5s: {{ row.measured25 }})</template>
          </text>
        </g>
      </svg>

      <p class="source-note">
        Source: <code>gomoku.scripts.bench_search</code> on two Neoverse-N1 cores (Oracle A1) at 2.5 s. Any value at <code>T ≠ 2.5 s</code> is a linear estimate (<code>T × N_2.5s / 2.5</code>), not a measurement.
      </p>
    </div>
  </div>
</template>

<script setup>
import { computed, ref } from 'vue';

const ROWS = [
  {
    id: '6x96-int8-2t',
    shortLabel: '6x96 · static int8 · 2T',
    label: '6x96 (v5) · static int8 ONNX · 2 threads',
    measured25: 634,
  },
  {
    id: '10x128-int8-2t',
    shortLabel: '10x128 · static int8 · 2T',
    label: '10x128 (v6b) · static int8 ONNX · 2 threads',
    measured25: 272,
  },
  {
    id: '6x96-int8-1t',
    shortLabel: '6x96 · static int8 · 1T',
    label: '6x96 (v5) · static int8 ONNX · 1 thread',
    measured25: 392,
  },
  {
    id: '6x96-fp32-2t',
    shortLabel: '6x96 · float32 · 2T',
    label: '6x96 (v5) · float32 ONNX · 2 threads',
    measured25: 196,
  },
];

const TIME_TICKS = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0];
const MAX_COUNT = 800;

const selectedId = ref('6x96-int8-2t');
const timeBudget = ref(2.5);

const activeRow = computed(
  () => ROWS.find((r) => r.id === selectedId.value) ?? ROWS[0]
);

const isExact = (val) => Math.abs(timeBudget.value - val) < 0.01;
const isMeasuredBudget = computed(() => isExact(2.5));

const activeRate = computed(() => activeRow.value.measured25 / 2.5);

function estimateFor(row) {
  if (isMeasuredBudget.value) return row.measured25;
  return Math.round((row.measured25 / 2.5) * timeBudget.value);
}

function timeToX(t) {
  return 44 + (t / 3.0) * 690;
}

function countToW(n) {
  return Math.min(430, Math.max(0, (n / MAX_COUNT) * 430));
}

function barColor(id) {
  if (id !== selectedId.value) return '#94a3b8';
  return isMeasuredBudget.value ? '#16a34a' : '#2563eb';
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

.controls-group {
  display: flex;
  flex-direction: column;
  gap: 10px;
  margin-bottom: 10px;
}

.row-buttons {
  display: flex;
  flex-wrap: wrap;
  justify-content: center;
  gap: 8px;
}

.cfg-btn,
.preset-btn {
  font-family: inherit;
  font-size: 12.5px;
  font-weight: 600;
  padding: 6px 11px;
  border: 1.5px solid #cbd5e1;
  border-radius: 6px;
  background: #ffffff;
  color: #334155;
  cursor: pointer;
  transition: all 0.15s ease;
}

.cfg-btn:hover,
.preset-btn:hover {
  background: #f1f5f9;
  border-color: #94a3b8;
}

.cfg-btn.active,
.preset-btn.active {
  background: #eff6ff;
  border-color: #2563eb;
  color: #1e40af;
}

.slider-row {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: center;
  gap: 10px;
}

.slider-label {
  font-size: 13px;
  color: #334155;
}

.range-input {
  width: 210px;
  max-width: 100%;
  accent-color: #2563eb;
  cursor: pointer;
}

.preset-btns {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}

.status-bar {
  background: #ffffff;
  border: 1.5px solid #e2e8f0;
  border-radius: 8px;
  padding: 10px 14px;
  margin-bottom: 10px;
  font-size: 13px;
  line-height: 1.45;
  color: #334155;
}

.status-bar.is-measured {
  background: #f0fdf4;
  border-color: #86efac;
}

.status-bar.is-over {
  background: #fffbeb;
  border-color: #fcd34d;
}

.status-main {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 8px;
  margin-bottom: 4px;
}

.badge {
  display: inline-block;
  font-size: 11.5px;
  font-weight: 700;
  padding: 2px 8px;
  border-radius: 999px;
}

.badge-green {
  background: #dcfce7;
  color: #166534;
  border: 1px solid #86efac;
}

.badge-amber {
  background: #fef3c7;
  color: #92400e;
  border: 1px solid #fcd34d;
}

.status-margin {
  font-size: 12.5px;
  color: #475569;
}

.diagram-svg {
  width: 100%;
  max-width: 100%;
  height: auto;
  display: block;
}

.bar-row {
  cursor: pointer;
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
