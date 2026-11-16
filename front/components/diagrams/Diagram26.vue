<template>
  <div class="diagram-wrapper">
    <div class="container">
      <h1>Model &amp; Backend Throughput on Two Neoverse-N1 Cores (2.5 s Budget)</h1>

      <div class="controls-group">
        <div class="filter-buttons" role="group" aria-label="Filter measurements">
          <button
            v-for="f in FILTERS"
            :key="f.id"
            type="button"
            class="filter-btn"
            :class="{ active: activeFilter === f.id }"
            @click="selectFilter(f.id)"
          >
            {{ f.label }}
          </button>
        </div>
      </div>

      <div class="status-bar" :class="activeRow.isProduction ? 'is-prod' : 'is-info'">
        <div class="status-main">
          <span class="badge" :class="badgeClass(activeRow)">
            {{ activeRow.badge }}
          </span>
          <span class="status-text">
            <strong>{{ activeRow.label }}</strong> —
            <strong>{{ activeRow.searches }} searches</strong> in 2.5 s
            (derived rate: {{ (activeRow.searches / 2.5).toFixed(1) }} / s,
            {{ (activeRow.searches / 196).toFixed(2) }}× vs. 6x96 float32)
          </span>
        </div>
        <div class="status-detail">
          {{ activeRow.note }}
        </div>
      </div>

      <svg
        class="diagram-svg"
        :viewBox="`0 0 760 ${svgHeight}`"
        xmlns="http://www.w3.org/2000/svg"
        role="img"
        aria-label="Bar chart of searches completed in 2.5 seconds on two Arm Neoverse-N1 cores"
      >
        <!-- Header row -->
        <text x="16" y="24" font-size="13.5" font-weight="700" fill="#0f172a">
          Configuration (click a row to inspect)
        </text>
        <text x="744" y="24" text-anchor="end" font-size="12" fill="#64748b">
          Searches in 2.5 s
        </text>

        <!-- Vertical grid lines -->
        <g v-for="tick in TICKS" :key="tick">
          <line
            :x1="barX(tick)"
            y1="36"
            :x2="barX(tick)"
            :y2="36 + visibleRows.length * ROW_H"
            stroke="#e2e8f0"
            stroke-width="1"
            stroke-dasharray="3 3"
          />
          <text
            :x="barX(tick)"
            :y="52 + visibleRows.length * ROW_H"
            text-anchor="middle"
            font-size="11.5"
            fill="#64748b"
          >
            {{ tick }}
          </text>
        </g>

        <!-- Rows -->
        <g
          v-for="(row, idx) in visibleRows"
          :key="row.id"
          class="bar-row"
          @click="selectedId = row.id"
        >
          <rect
            x="8"
            :y="40 + idx * ROW_H"
            width="744"
            :height="ROW_H - 6"
            rx="6"
            :fill="selectedId === row.id ? '#eff6ff' : 'transparent'"
            :stroke="selectedId === row.id ? '#93c5fd' : 'transparent'"
            stroke-width="1.5"
          />

          <!-- Label & source tag -->
          <text
            x="16"
            :y="60 + idx * ROW_H"
            font-size="13"
            :font-weight="selectedId === row.id || row.isProduction ? '700' : '600'"
            :fill="selectedId === row.id ? '#1e3a8a' : '#1e293b'"
          >
            {{ row.shortLabel }}
          </text>
          <text
            x="16"
            :y="76 + idx * ROW_H"
            font-size="11"
            fill="#64748b"
          >
            {{ row.sourceTag }}
          </text>

          <!-- Track -->
          <rect
            :x="BAR_LEFT"
            :y="50 + idx * ROW_H"
            :width="BAR_WIDTH"
            height="24"
            rx="4"
            fill="#f1f5f9"
            stroke="#e2e8f0"
            stroke-width="1"
          />

          <!-- Bar -->
          <rect
            :x="BAR_LEFT"
            :y="50 + idx * ROW_H"
            :width="barW(row.searches)"
            height="24"
            rx="4"
            :fill="barColor(row)"
          />

          <!-- Value -->
          <text
            :x="BAR_LEFT + BAR_WIDTH + 12"
            :y="67 + idx * ROW_H"
            font-size="13.5"
            font-weight="700"
            :fill="row.isProduction ? '#15803d' : '#0f172a'"
          >
            {{ row.searches }}
          </text>
        </g>

        <!-- Axis caption -->
        <text
          :x="BAR_LEFT + BAR_WIDTH / 2"
          :y="72 + visibleRows.length * ROW_H"
          text-anchor="middle"
          font-size="12"
          font-weight="600"
          fill="#475569"
        >
          Simulations completed in 2.5 s (time_limit_s: 2.5)
        </text>
      </svg>

      <div class="legend">
        <span class="legend-item">
          <span class="swatch swatch-green"></span>
          Deployed choice (6x96 · static int8 · 2T)
        </span>
        <span class="legend-item">
          <span class="swatch swatch-blue"></span>
          Selected row
        </span>
        <span class="legend-item">
          <span class="swatch swatch-slate"></span>
          Other measured configuration
        </span>
      </div>

      <p class="source-note">
        Sources: <code>deploy/alphazero-oracle/README.md</code> pre-deploy table (<code>634</code>, <code>272</code>, <code>392</code>, <code>196</code> via <code>gomoku.scripts.bench_search</code>) and on-instance <code>03_bench_search.sh</code> runs for v5 iter 144 (<code>613</code> as trained, <code>671</code> pruned). Throughput rates (<code>N / 2.5</code>) are derived from those counts.
      </p>
    </div>
  </div>
</template>

<script setup>
import { computed, ref } from 'vue';

const ALL_ROWS = [
  {
    id: 'v5-pruned-onhost',
    shortLabel: '6x96 (v5 iter 144 pruned) · int8 · 2T',
    label: '6x96 (v5 iter 144, pruned 440/576 ch) · static int8 ONNX · 2 threads',
    searches: 671,
    category: 'model',
    isProduction: true,
    badge: 'Deployed (pruned)',
    sourceTag: 'On-instance: 03_bench_search.sh',
    note: 'Removing 136 zeroed channels raised on-instance 2.5 s throughput from 613 to 671 searches (configs/deploy-v5-pruned.yaml).',
  },
  {
    id: 'v5-predeploy-2t',
    shortLabel: '6x96 (v5) · static int8 · 2T',
    label: '6x96 (v5) · static int8 ONNX · 2 threads',
    searches: 634,
    category: 'both',
    isProduction: true,
    badge: 'Chosen default',
    sourceTag: 'Pre-deploy: bench_search',
    note: '2.33× as many searches as 10x128 (272) and 3.23× as many as float32 (196) on the same two Neoverse-N1 cores.',
  },
  {
    id: 'v5-trained-onhost',
    shortLabel: '6x96 (v5 iter 144) · int8 · 2T',
    label: '6x96 (v5 iter 144, as trained) · static int8 ONNX · 2 threads',
    searches: 613,
    category: 'model',
    isProduction: false,
    badge: 'On-instance baseline',
    sourceTag: 'On-instance: 03_bench_search.sh',
    note: 'Measured inside the live container image with STOP_ENGINE=true before channel pruning (configs/deploy-v5.yaml).',
  },
  {
    id: 'v5-predeploy-1t',
    shortLabel: '6x96 (v5) · static int8 · 1T',
    label: '6x96 (v5) · static int8 ONNX · 1 thread',
    searches: 392,
    category: 'scaling',
    isProduction: false,
    badge: 'Thread scaling',
    sourceTag: 'Pre-deploy: bench_search',
    note: 'Using 2 threads instead of 1 raises 2.5 s search count from 392 to 634 (+61.7%), so the instance dedicates both cores.',
  },
  {
    id: 'v6b-predeploy-2t',
    shortLabel: '10x128 (v6b) · static int8 · 2T',
    label: '10x128 (v6b) · static int8 ONNX · 2 threads',
    searches: 272,
    category: 'model',
    isProduction: false,
    badge: 'Not deployed',
    sourceTag: 'Pre-deploy: bench_search',
    note: 'More capacity per node evaluation, but completes only 272 searches in 2.5 s on A1 and scored 0.46–0.50 against v5 at equal time.',
  },
  {
    id: 'v5-fp32-2t',
    shortLabel: '6x96 (v5) · float32 ONNX · 2T',
    label: '6x96 (v5) · float32 ONNX · 2 threads',
    searches: 196,
    category: 'scaling',
    isProduction: false,
    badge: 'Backend baseline',
    sourceTag: 'Pre-deploy: bench_search',
    note: 'Unquantized float32 ONNX completes fewer than 200 searches in 2.5 s; static int8 lifts the same 6x96 network by 3.23×.',
  },
];

const FILTERS = [
  { id: 'all', label: 'All Measured (6)' },
  { id: 'model', label: 'Model Size & Pruning' },
  { id: 'scaling', label: 'Backend & Threads (6x96)' },
];

const BAR_LEFT = 278;
const BAR_WIDTH = 412;
const MAX_SEARCHES = 700;
const ROW_H = 50;
const TICKS = [0, 200, 400, 600, 700];

const activeFilter = ref('all');
const selectedId = ref('v5-pruned-onhost');

const visibleRows = computed(() => {
  if (activeFilter.value === 'all') return ALL_ROWS;
  if (activeFilter.value === 'model') {
    return ALL_ROWS.filter((r) => r.category === 'model' || r.category === 'both');
  }
  return ALL_ROWS.filter((r) => r.category === 'scaling' || r.category === 'both');
});

const svgHeight = computed(() => 84 + visibleRows.value.length * ROW_H);

const activeRow = computed(
  () => ALL_ROWS.find((r) => r.id === selectedId.value) ?? visibleRows.value[0]
);

function selectFilter(id) {
  activeFilter.value = id;
  if (!visibleRows.value.some((r) => r.id === selectedId.value)) {
    selectedId.value = visibleRows.value[0].id;
  }
}

function barW(n) {
  return Math.max(0, Math.min(BAR_WIDTH, (n / MAX_SEARCHES) * BAR_WIDTH));
}

function barX(tick) {
  return BAR_LEFT + barW(tick);
}

function barColor(row) {
  if (row.id === selectedId.value) {
    return row.isProduction ? '#15803d' : '#2563eb';
  }
  return row.isProduction ? '#22c55e' : '#94a3b8';
}

function badgeClass(row) {
  return row.isProduction ? 'badge-green' : 'badge-slate';
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
  justify-content: center;
  margin-bottom: 10px;
}

.filter-buttons {
  display: flex;
  flex-wrap: wrap;
  justify-content: center;
  gap: 8px;
}

.filter-btn {
  font-family: inherit;
  font-size: 12.5px;
  font-weight: 600;
  padding: 6px 12px;
  border: 1.5px solid #cbd5e1;
  border-radius: 6px;
  background: #ffffff;
  color: #334155;
  cursor: pointer;
  transition: all 0.15s ease;
}

.filter-btn:hover {
  background: #f1f5f9;
  border-color: #94a3b8;
}

.filter-btn.active {
  background: #eff6ff;
  border-color: #2563eb;
  color: #1e40af;
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

.status-bar.is-prod {
  background: #f0fdf4;
  border-color: #86efac;
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

.badge-slate {
  background: #e2e8f0;
  color: #334155;
  border: 1px solid #cbd5e1;
}

.status-detail {
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

.legend {
  display: flex;
  flex-wrap: wrap;
  justify-content: center;
  gap: 16px;
  margin-top: 8px;
  font-size: 12.5px;
  color: #475569;
}

.legend-item {
  display: flex;
  align-items: center;
  gap: 6px;
}

.swatch {
  display: inline-block;
  width: 14px;
  height: 14px;
  border-radius: 3px;
}

.swatch-green { background: #15803d; }
.swatch-blue { background: #2563eb; }
.swatch-slate { background: #94a3b8; }

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
