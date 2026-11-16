<template>
  <div class="diagram-wrapper">
    <div class="container">
      <h1>Equal Search Counts vs. Equal Thinking Time (v6b vs. v5 iter 144)</h1>

      <div class="controls-group" role="group" aria-label="Comparison regime">
        <button
          v-for="mode in MODES"
          :key="mode.id"
          type="button"
          class="mode-btn"
          :class="{ active: activeId === mode.id }"
          @click="activeId = mode.id"
        >
          {{ mode.buttonLabel }}
        </button>
      </div>

      <div class="status-bar" :class="activeMode.verdictClass">
        <div class="status-main">
          <span class="badge" :class="activeMode.badgeClass">{{ activeMode.badgeText }}</span>
          <span class="status-text">
            <strong>{{ activeMode.title }}</strong> — {{ activeMode.summary }}
          </span>
        </div>
        <div class="status-sub">{{ activeMode.verdict }}</div>
      </div>

      <svg
        class="diagram-svg"
        viewBox="0 0 780 365"
        xmlns="http://www.w3.org/2000/svg"
        role="img"
        aria-label="Search budgets and measured match scores between v6b and v5 iteration 144"
      >
        <!-- Panel 1: Search budget per move -->
        <text x="24" y="24" font-size="14" font-weight="700" fill="#0f172a">
          1. Simulations per Move in Match
        </text>
        <text x="756" y="24" text-anchor="end" font-size="11.5" fill="#64748b">
          {{ activeMode.budgetNote }}
        </text>

        <!-- v6b bar -->
        <text x="24" y="58" font-size="13" font-weight="700" fill="#1e3a8a">v6b (10×128)</text>
        <rect x="175" y="42" width="460" height="24" rx="4" fill="#f1f5f9" stroke="#e2e8f0" stroke-width="1" />
        <rect x="175" y="42" :width="searchToW(activeMode.v6bSearches)" height="24" rx="4" fill="#2563eb" />
        <text x="645" y="59" font-size="13" font-weight="700" fill="#0f172a">
          {{ formatInt(activeMode.v6bSearches) }} searches
        </text>

        <!-- v5 bar -->
        <text x="24" y="94" font-size="13" font-weight="700" fill="#166534">v5 iter 144 (6×96)</text>
        <rect x="175" y="78" width="460" height="24" rx="4" fill="#f1f5f9" stroke="#e2e8f0" stroke-width="1" />
        <rect x="175" y="78" :width="searchToW(activeMode.v5Searches)" height="24" rx="4" fill="#16a34a" />
        <text x="645" y="95" font-size="13" font-weight="700" fill="#0f172a">
          {{ formatInt(activeMode.v5Searches) }} searches
        </text>

        <!-- Divider -->
        <line x1="24" y1="122" x2="756" y2="122" stroke="#e2e8f0" stroke-width="1.5" />

        <!-- Panel 2: Measured match score axis (0.42 to 0.58) -->
        <text x="24" y="148" font-size="14" font-weight="700" fill="#0f172a">
          2. Measured Score of v6b vs. v5 Iteration 144 (3 random opening plies)
        </text>
        <text x="756" y="148" text-anchor="end" font-size="11.5" fill="#64748b">
          Draws count 0.5 · 0.500 = even
        </text>

        <!-- Left / right background tint around 0.500 -->
        <rect x="175" y="164" :width="scoreToX(0.5) - 175" height="148" fill="#fef2f2" opacity="0.55" />
        <rect :x="scoreToX(0.5)" y="164" :width="685 - scoreToX(0.5)" height="148" fill="#f0fdf4" opacity="0.6" />

        <text :x="(175 + scoreToX(0.5)) / 2" y="180" text-anchor="middle" font-size="11.5" font-weight="700" fill="#b91c1c">
          v5 (6×96) ahead (&lt; 0.500)
        </text>
        <text :x="(scoreToX(0.5) + 685) / 2" y="180" text-anchor="middle" font-size="11.5" font-weight="700" fill="#15803d">
          v6b (10×128) ahead (&gt; 0.500)
        </text>

        <!-- Vertical 0.500 reference line -->
        <line :x1="scoreToX(0.5)" y1="164" :x2="scoreToX(0.5)" y2="316" stroke="#0f172a" stroke-width="2" stroke-dasharray="5 4" />

        <!-- Grid ticks -->
        <g v-for="t in SCORE_TICKS" :key="t">
          <line :x1="scoreToX(t)" y1="186" :x2="scoreToX(t)" y2="312" stroke="#cbd5e1" stroke-width="1" opacity="0.7" />
          <text :x="scoreToX(t)" y="332" text-anchor="middle" font-size="11.5" fill="#475569">
            {{ t.toFixed(2) }}
          </text>
        </g>

        <!-- Measured points for the active mode -->
        <g v-for="(pt, idx) in activeMode.points" :key="pt.label">
          <text
            x="165"
            :y="rowY(idx) + 4"
            text-anchor="end"
            font-size="12"
            font-weight="600"
            fill="#334155"
          >
            {{ pt.label }}
          </text>
          <line
            x1="175"
            :y1="rowY(idx)"
            x2="685"
            :y2="rowY(idx)"
            stroke="#e2e8f0"
            stroke-width="1"
          />
          <line
            :x1="scoreToX(0.5)"
            :y1="rowY(idx)"
            :x2="scoreToX(pt.score)"
            :y2="rowY(idx)"
            :stroke="pt.score >= 0.5 ? '#16a34a' : '#dc2626'"
            stroke-width="3"
          />
          <circle
            :cx="scoreToX(pt.score)"
            :cy="rowY(idx)"
            r="6"
            :fill="pt.score >= 0.5 ? '#16a34a' : '#dc2626'"
            stroke="#ffffff"
            stroke-width="1.5"
          />
          <text
            :x="pt.score >= 0.5 ? scoreToX(pt.score) + 10 : scoreToX(pt.score) - 10"
            :y="rowY(idx) + 4"
            :text-anchor="pt.score >= 0.5 ? 'start' : 'end'"
            font-size="12"
            font-weight="700"
            :fill="pt.score >= 0.5 ? '#166534' : '#991b1b'"
          >
            {{ pt.score.toFixed(3) }}
          </text>
        </g>
      </svg>

      <p class="source-note">
        Sources: <code>v6b-10x128.yaml</code>, <code>v6b-10x128-cont2.yaml</code>, <code>deploy-v6b.yaml</code>, and GitHub issue #49 (all matches vs. v5 iteration 144 with 3 random opening plies).
      </p>
    </div>
  </div>
</template>

<script setup>
import { computed, ref } from 'vue';

const MODES = [
  {
    id: 'arm',
    buttonLabel: 'Equal Time · Arm A1 (270 vs 630)',
    title: 'Equal thinking time on Oracle A1 (2.5 s static int8, 2 threads)',
    summary: '270 searches for v6b vs. 630 for v5 across 300 games per match.',
    budgetNote: 'A1 bench_search: 272 vs. 634 (pruned v5: 671)',
    verdict: 'Deployment verdict: v6b never crossed 0.500 (0.460–0.497), so v5 stayed in production.',
    badgeText: 'Decides deployment',
    badgeClass: 'badge-red',
    verdictClass: 'is-behind',
    v6bSearches: 270,
    v5Searches: 630,
    points: [
      { label: 'iter 13 (lr 3e-4)', score: 0.480 },
      { label: 'iter 16 (lr 1e-4)', score: 0.460 },
      { label: 'iter 20 (lr 1e-4)', score: 0.480 },
      { label: 'iter 24 (lr 1e-4)', score: 0.497 },
      { label: 'iter 28 (lr 5e-5)', score: 0.490 },
      { label: 'iter 32 (lr 5e-5)', score: 0.487 },
    ],
  },
  {
    id: 'c4d',
    buttonLabel: 'Equal Time · C4D Ratio (640 vs 1,600)',
    title: 'Initial 2.5× C4D worker ratio (bfloat16, batch 8: 39 ms vs. 16 ms)',
    summary: '640 searches for v6b vs. 1,600 for v5 across 200 games per match (iters 4–12).',
    budgetNote: 'Initial ratio before Arm A1 measurements',
    verdict: 'Result: flat at 0.460–0.475 under lr = 0.0003, prompting the switch to 270 vs. 630 and early LR decay.',
    badgeText: 'Pre-Arm equal time',
    badgeClass: 'badge-amber',
    verdictClass: 'is-caution',
    v6bSearches: 640,
    v5Searches: 1600,
    points: [
      { label: 'iter 4 (lr 3e-4)', score: 0.465 },
      { label: 'iter 8 (lr 3e-4)', score: 0.475 },
      { label: 'iter 12 (lr 3e-4)', score: 0.460 },
    ],
  },
  {
    id: 'equal-searches',
    buttonLabel: 'Equal Searches (1,600 vs 1,600)',
    title: 'Equal simulation count (ignoring inference cost)',
    summary: '1,600 searches each at v6b iteration 12.',
    budgetNote: 'Measures per-search strength, not 2.5 s serving strength',
    verdict: 'Result: v6b wins at 0.540 (+28 Elo per search), showing capacity helps per simulation even though Arm throughput reverses it.',
    badgeText: 'Favours bigger net',
    badgeClass: 'badge-green',
    verdictClass: 'is-ahead',
    v6bSearches: 1600,
    v5Searches: 1600,
    points: [{ label: 'iter 12 (1600 each)', score: 0.540 }],
  },
];

const SCORE_TICKS = [0.42, 0.44, 0.46, 0.48, 0.5, 0.52, 0.54, 0.56];
const activeId = ref('arm');

const activeMode = computed(() => MODES.find((m) => m.id === activeId.value) ?? MODES[0]);

function searchToW(n) {
  return Math.max(4, (n / 1600) * 460);
}

function scoreToX(s) {
  return 175 + ((s - 0.42) / (0.58 - 0.42)) * 510;
}

function rowY(idx) {
  const count = activeMode.value.points.length;
  if (count === 1) return 245;
  if (count === 3) return 212 + idx * 36;
  return 198 + idx * 21;
}

function formatInt(n) {
  return Number(n).toLocaleString('en-US');
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
  flex-wrap: wrap;
  justify-content: center;
  gap: 8px;
  margin-bottom: 10px;
}

.mode-btn {
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

.mode-btn:hover {
  background: #f1f5f9;
  border-color: #94a3b8;
}

.mode-btn.active {
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

.status-bar.is-ahead {
  background: #f0fdf4;
  border-color: #86efac;
}

.status-bar.is-behind {
  background: #fef2f2;
  border-color: #fca5a5;
}

.status-bar.is-caution {
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

.badge-red {
  background: #fee2e2;
  color: #991b1b;
  border: 1px solid #fca5a5;
}

.badge-amber {
  background: #fef3c7;
  color: #92400e;
  border: 1px solid #fcd34d;
}

.status-sub {
  font-size: 12.5px;
  color: #475569;
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
