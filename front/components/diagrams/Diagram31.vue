<template>
  <div class="diagram-wrapper">
    <div class="container">
      <h1>Oracle A1 Full-Game Measurements (<code>v5</code> iter 144 pruned, 2 threads)</h1>

      <div class="controls-panel">
        <div class="metric-tabs" role="tablist" aria-label="Measured metric">
          <button
            v-for="tab in TABS"
            :key="tab.id"
            type="button"
            role="tab"
            :aria-selected="activeTab === tab.id"
            class="tab-btn"
            :class="{ active: activeTab === tab.id }"
            @click="activeTab = tab.id"
          >
            {{ tab.label }}
          </button>
        </div>

        <div class="row-pills" role="group" aria-label="Select condition">
          <button
            v-for="row in ROWS"
            :key="row.id"
            type="button"
            class="pill-btn"
            :class="{ active: selectedId === row.id }"
            @click="selectedId = row.id"
          >
            {{ row.shortLabel }}
          </button>
        </div>
      </div>

      <div class="detail-card">
        <div class="detail-title">
          <strong>{{ activeRow.label }}</strong>
          <span class="badge-note">1 self-play game · --connections 2 · medians after 1st move</span>
        </div>
        <div class="detail-grid">
          <div class="stat-cell">
            <span class="stat-k">Behind move (median)</span>
            <span class="stat-v">{{ activeRow.total.toLocaleString() }}</span>
          </div>
          <div class="stat-cell">
            <span class="stat-k">Carried over / New</span>
            <span class="stat-v">{{ activeRow.carried.toLocaleString() }} / {{ activeRow.newCount.toLocaleString() }}</span>
          </div>
          <div class="stat-cell">
            <span class="stat-k">Answer mean / max</span>
            <span class="stat-v">{{ activeRow.meanS.toFixed(2) }} s / {{ activeRow.maxS.toFixed(2) }} s</span>
          </div>
          <div class="stat-cell">
            <span class="stat-k">Container memory peak</span>
            <span class="stat-v">{{ activeRow.memMb }} MB</span>
          </div>
        </div>
      </div>

      <svg
        class="diagram-svg"
        viewBox="0 0 780 286"
        xmlns="http://www.w3.org/2000/svg"
        role="img"
        aria-label="Bar chart of Oracle A1 full-game measurements across tree reuse and pondering settings"
      >
        <!-- Header & Legend -->
        <template v-if="activeTab === 'searches'">
          <text x="20" y="22" font-size="13.5" font-weight="700" fill="#0f172a">
            Median Searches Behind a Move (Carried Over vs. New)
          </text>
          <rect x="492" y="10" width="12" height="12" rx="2" fill="#16a34a" />
          <text x="509" y="21" font-size="11.5" fill="#334155">Carried over</text>
          <rect x="594" y="10" width="12" height="12" rx="2" fill="#2563eb" />
          <text x="611" y="21" font-size="11.5" fill="#334155">New</text>
        </template>
        <template v-else-if="activeTab === 'latency'">
          <text x="20" y="22" font-size="13.5" font-weight="700" fill="#0f172a">
            Answer Latency vs. 2.5 s Search Limit and 3.0 s Target
          </text>
          <rect x="500" y="10" width="12" height="12" rx="2" fill="#2563eb" />
          <text x="517" y="21" font-size="11.5" fill="#334155">Mean</text>
          <line x1="566" y1="8" x2="566" y2="24" stroke="#dc2626" stroke-width="2.5" />
          <text x="574" y="21" font-size="11.5" fill="#334155">Max</text>
        </template>
        <template v-else>
          <text x="20" y="22" font-size="13.5" font-weight="700" fill="#0f172a">
            Container Memory Peak (MB)
          </text>
          <line x1="470" y1="8" x2="470" y2="24" stroke="#d97706" stroke-width="2" stroke-dasharray="4 3" />
          <text x="478" y="21" font-size="11.5" fill="#92400e">461 MB (130 s cap check peak)</text>
        </template>

        <!-- Vertical reference lines for latency and memory -->
        <template v-if="activeTab === 'latency'">
          <line :x1="latX(2.5)" y1="34" :x2="latX(2.5)" y2="246" stroke="#15803d" stroke-width="1.5" stroke-dasharray="4 3" />
          <text :x="latX(2.5)" y="262" text-anchor="middle" font-size="11" font-weight="700" fill="#15803d">2.5 s limit</text>
          <line :x1="latX(3.0)" y1="34" :x2="latX(3.0)" y2="246" stroke="#dc2626" stroke-width="1.5" stroke-dasharray="4 3" />
          <text :x="latX(3.0)" y="262" text-anchor="middle" font-size="11" font-weight="700" fill="#dc2626">3.0 s target</text>
        </template>

        <template v-if="activeTab === 'memory'">
          <line :x1="memX(461)" y1="34" :x2="memX(461)" y2="246" stroke="#d97706" stroke-width="1.5" stroke-dasharray="4 3" />
          <text :x="memX(461)" y="262" text-anchor="middle" font-size="11" font-weight="700" fill="#92400e">461 MB (130 s)</text>
        </template>

        <!-- Rows -->
        <g
          v-for="(row, idx) in ROWS"
          :key="row.id"
          class="chart-row"
          @click="selectedId = row.id"
        >
          <rect
            x="12"
            :y="36 + idx * 42"
            width="756"
            height="36"
            rx="6"
            :fill="selectedId === row.id ? '#eff6ff' : 'transparent'"
            :stroke="selectedId === row.id ? '#93c5fd' : 'transparent'"
            stroke-width="1.5"
          />

          <text
            x="22"
            :y="58 + idx * 42"
            font-size="12.5"
            :font-weight="selectedId === row.id ? '700' : '600'"
            :fill="selectedId === row.id ? '#1e3a8a' : '#334155'"
          >
            {{ row.shortLabel }}
          </text>

          <!-- Track -->
          <rect x="208" :y="44 + idx * 42" width="420" height="20" rx="4" fill="#f1f5f9" stroke="#e2e8f0" stroke-width="1" />

          <!-- Metric 1: Searches -->
          <template v-if="activeTab === 'searches'">
            <rect
              v-if="row.carried > 0"
              x="208"
              :y="44 + idx * 42"
              :width="searchW(row.carried)"
              height="20"
              rx="3"
              fill="#16a34a"
            />
            <rect
              :x="208 + searchW(row.carried)"
              :y="44 + idx * 42"
              :width="Math.max(3, searchW(row.newCount))"
              height="20"
              rx="3"
              fill="#2563eb"
            />
            <text
              x="638"
              :y="58 + idx * 42"
              font-size="12"
              :font-weight="selectedId === row.id ? '700' : '600'"
              fill="#0f172a"
            >
              {{ row.total.toLocaleString() }} ({{ row.carried.toLocaleString() }} + {{ row.newCount.toLocaleString() }})
            </text>
          </template>

          <!-- Metric 2: Latency -->
          <template v-else-if="activeTab === 'latency'">
            <rect
              x="208"
              :y="44 + idx * 42"
              :width="latW(row.meanS)"
              height="20"
              rx="4"
              fill="#2563eb"
              opacity="0.88"
            />
            <line
              :x1="latX(row.maxS)"
              :y1="40 + idx * 42"
              :x2="latX(row.maxS)"
              :y2="68 + idx * 42"
              stroke="#dc2626"
              stroke-width="2.5"
            />
            <circle :cx="latX(row.maxS)" :cy="54 + idx * 42" r="3.5" fill="#dc2626" />
            <text
              x="638"
              :y="58 + idx * 42"
              font-size="12"
              :font-weight="selectedId === row.id ? '700' : '600'"
              fill="#0f172a"
            >
              {{ row.meanS.toFixed(2) }} s / {{ row.maxS.toFixed(2) }} s
            </text>
          </template>

          <!-- Metric 3: Memory -->
          <template v-else>
            <rect
              x="208"
              :y="44 + idx * 42"
              :width="memW(row.memMb)"
              height="20"
              rx="4"
              :fill="row.memMb > 300 ? '#0284c7' : '#64748b'"
            />
            <text
              x="638"
              :y="58 + idx * 42"
              font-size="12"
              :font-weight="selectedId === row.id ? '700' : '600'"
              fill="#0f172a"
            >
              {{ row.memMb }} MB
            </text>
          </template>
        </g>

        <!-- Footnote inside SVG -->
        <text v-if="activeTab === 'searches'" x="20" y="274" font-size="11" fill="#64748b">
          Each column is a separate median across moves after a connection's first, so median(total) differs slightly from median(carried) + median(new).
        </text>
        <text v-else-if="activeTab === 'latency'" x="20" y="278" font-size="11" fill="#64748b">
          All configurations stayed below 2.60 s max, inside the 3.00 s move target.
        </text>
        <text v-else x="20" y="278" font-size="11" fill="#64748b">
          Shared cap is 50,000 nodes (4–5 KB/node); freed connection memory is reused for later trees, not returned to the OS.
        </text>
      </svg>

      <p class="source-note">
        Source: <code>deploy/alphazero-oracle/README.md</code>. Self-play opponent replies match the kept tree more often than a human opponent, so carried-over counts are an upper range.
      </p>
    </div>
  </div>
</template>

<script setup>
import { computed, ref } from 'vue';

const TABS = [
  { id: 'searches', label: 'Searches per move' },
  { id: 'latency', label: 'Answer latency' },
  { id: 'memory', label: 'Container memory' },
];

const ROWS = [
  {
    id: 'none',
    shortLabel: 'No kept tree',
    label: 'No kept tree (03_bench_search.sh median / fresh tree per request)',
    total: 668,
    carried: 0,
    newCount: 668,
    meanS: 2.52,
    maxS: 2.59,
    memMb: 262,
  },
  {
    id: 'reuse',
    shortLabel: 'reuse_tree',
    label: 'reuse_tree only',
    total: 1468,
    carried: 646,
    newCount: 812,
    meanS: 2.42,
    maxS: 2.53,
    memMb: 262,
  },
  {
    id: 'ponder-0',
    shortLabel: 'reuse + ponder (0 s)',
    label: 'reuse_tree + ponder, opponent answers at once',
    total: 1468,
    carried: 651,
    newCount: 799,
    meanS: 2.42,
    maxS: 2.56,
    memMb: 263,
  },
  {
    id: 'ponder-5',
    shortLabel: 'reuse + ponder (5 s)',
    label: 'reuse_tree + ponder, opponent thinks 5 s',
    total: 6363,
    carried: 5303,
    newCount: 1044,
    meanS: 2.51,
    maxS: 2.53,
    memMb: 307,
  },
  {
    id: 'ponder-15',
    shortLabel: 'reuse + ponder (15 s)',
    label: 'reuse_tree + ponder, opponent thinks 15 s',
    total: 13859,
    carried: 12484,
    newCount: 977,
    meanS: 2.51,
    maxS: 2.51,
    memMb: 415,
  },
];

const activeTab = ref('searches');
const selectedId = ref('ponder-5');

const activeRow = computed(
  () => ROWS.find((r) => r.id === selectedId.value) ?? ROWS[0]
);

const MAX_SEARCHES = 14500;
const MAX_LATENCY = 3.2;
const MAX_MEMORY = 500;

function searchW(n) {
  return Math.max(0, Math.min(420, (n / MAX_SEARCHES) * 420));
}

function latW(s) {
  return Math.max(0, Math.min(420, (s / MAX_LATENCY) * 420));
}

function latX(s) {
  return 208 + latW(s);
}

function memW(mb) {
  return Math.max(0, Math.min(420, (mb / MAX_MEMORY) * 420));
}

function memX(mb) {
  return 208 + memW(mb);
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

.metric-tabs,
.row-pills {
  display: flex;
  flex-wrap: wrap;
  justify-content: center;
  gap: 8px;
}

.tab-btn,
.pill-btn {
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

.tab-btn:hover,
.pill-btn:hover {
  background: #f1f5f9;
  border-color: #94a3b8;
}

.tab-btn.active {
  background: #0f172a;
  border-color: #0f172a;
  color: #ffffff;
}

.pill-btn.active {
  background: #eff6ff;
  border-color: #2563eb;
  color: #1e40af;
}

.detail-card {
  background: #ffffff;
  border: 1.5px solid #e2e8f0;
  border-radius: 8px;
  padding: 10px 14px;
  margin-bottom: 10px;
}

.detail-title {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: space-between;
  gap: 6px;
  margin-bottom: 8px;
  font-size: 13px;
  color: #0f172a;
}

.badge-note {
  font-size: 11px;
  font-weight: 600;
  padding: 2px 8px;
  border-radius: 999px;
  background: #f1f5f9;
  color: #475569;
  border: 1px solid #cbd5e1;
}

.detail-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 8px;
}

.stat-cell {
  background: #f8fafc;
  border: 1px solid #e2e8f0;
  border-radius: 6px;
  padding: 6px 8px;
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.stat-k {
  font-size: 11px;
  color: #64748b;
}

.stat-v {
  font-size: 13px;
  font-weight: 700;
  color: #0f172a;
}

.diagram-svg {
  width: 100%;
  max-width: 100%;
  height: auto;
  display: block;
}

.chart-row {
  cursor: pointer;
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
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}

@media (max-width: 480px) {
  h1 {
    font-size: 16.5px;
  }
}
</style>
