<template>
  <div class="diagram-wrapper">
    <div class="container">
      <h1>v6b Trajectory Over Iterations 0–32 (Stopped Before 50-Iter Target)</h1>

      <div class="controls-group">
        <div class="btn-row" role="group" aria-label="Series filter">
          <button
            v-for="opt in SERIES_OPTIONS"
            :key="opt.id"
            type="button"
            class="ctrl-btn"
            :class="{ active: seriesFilter === opt.id }"
            @click="seriesFilter = opt.id"
          >
            {{ opt.label }}
          </button>
          <button
            type="button"
            class="ctrl-btn"
            :class="{ active: showC4dSeries }"
            @click="showC4dSeries = !showC4dSeries"
          >
            {{ showC4dSeries ? 'Hide' : 'Show' }} iter 4–12 (640 vs 1,600)
          </button>
        </div>
      </div>

      <div class="status-bar">
        <div class="status-main">
          <span class="badge" :class="selectedBadgeClass">{{ selectedPoint.seriesLabel }}</span>
          <span class="status-text">
            <strong>Iteration {{ selectedPoint.iter }}</strong>:
            score <strong>{{ selectedPoint.score.toFixed(3) }}</strong>
            ({{ (selectedPoint.score * 100).toFixed(1) }}%) ·
            learning rate <code>{{ selectedPoint.lr }}</code> ·
            {{ selectedPoint.detail }}
          </span>
        </div>
        <div class="status-sub">
          Click any point on the chart to inspect it. Training stopped at iteration 32 for time and budget while <code>v6b-10x128-cont2.yaml</code> ran to 50.
        </div>
      </div>

      <svg
        class="diagram-svg"
        viewBox="0 0 780 390"
        xmlns="http://www.w3.org/2000/svg"
        role="img"
        aria-label="v6b win rate trajectory across iterations 0 to 50 showing cutoff at iteration 32"
      >
        <!-- LR schedule shaded zones across plot area (x: 68..748, y: 46..312) -->
        <rect :x="iterToX(0)" y="46" :width="iterToX(14) - iterToX(0)" height="266" fill="#f8fafc" />
        <rect :x="iterToX(14)" y="46" :width="iterToX(24) - iterToX(14)" height="266" fill="#eff6ff" opacity="0.65" />
        <rect :x="iterToX(24)" y="46" :width="iterToX(32) - iterToX(24)" height="266" fill="#eef2ff" opacity="0.7" />
        <rect :x="iterToX(32)" y="46" :width="iterToX(50) - iterToX(32)" height="266" fill="#fef3c7" opacity="0.45" />

        <!-- LR zone labels at top -->
        <text :x="(iterToX(0) + iterToX(14)) / 2" y="36" text-anchor="middle" font-size="11" font-weight="600" fill="#64748b">
          lr 3e-4 (iters 1–14)
        </text>
        <text :x="(iterToX(14) + iterToX(24)) / 2" y="36" text-anchor="middle" font-size="11" font-weight="700" fill="#1d4ed8">
          lr 1e-4 (15–24)
        </text>
        <text :x="(iterToX(24) + iterToX(32)) / 2" y="36" text-anchor="middle" font-size="11" font-weight="700" fill="#4338ca">
          5e-5 (25–32)
        </text>
        <text :x="(iterToX(32) + iterToX(50)) / 2" y="36" text-anchor="middle" font-size="11" font-weight="700" fill="#92400e">
          Unrun (33–50: 5e-5 → 2e-5)
        </text>

        <!-- Plot frame -->
        <rect x="68" y="46" width="680" height="266" fill="none" stroke="#cbd5e1" stroke-width="1.5" />

        <!-- Y grid lines and labels -->
        <g v-for="yVal in Y_TICKS" :key="yVal">
          <line
            x1="68"
            :y1="scoreToY(yVal)"
            x2="748"
            :y2="scoreToY(yVal)"
            :stroke="isHalf(yVal) ? '#0f172a' : '#e2e8f0'"
            :stroke-width="isHalf(yVal) ? '2' : '1'"
            :stroke-dasharray="isHalf(yVal) ? '6 4' : 'none'"
          />
          <text
            x="60"
            :y="scoreToY(yVal) + 4"
            text-anchor="end"
            font-size="11.5"
            :font-weight="isHalf(yVal) ? '700' : '500'"
            :fill="isHalf(yVal) ? '#0f172a' : '#64748b'"
          >
            {{ yVal.toFixed(2) }}
          </text>
        </g>

        <text :x="742" :y="scoreToY(0.5) - 7" text-anchor="end" font-size="11.5" font-weight="700" fill="#0f172a">
          0.500 break-even vs. opponent
        </text>

        <!-- X ticks -->
        <g v-for="it in X_TICKS" :key="it">
          <line :x1="iterToX(it)" y1="312" :x2="iterToX(it)" y2="318" stroke="#64748b" stroke-width="1.5" />
          <text :x="iterToX(it)" y="333" text-anchor="middle" font-size="11.5" fill="#475569">
            {{ it }}
          </text>
        </g>
        <text x="408" y="352" text-anchor="middle" font-size="12" font-weight="600" fill="#475569">
          v6b Training Iteration
        </text>

        <!-- Early LR drop marker at iter 15 -->
        <line :x1="iterToX(15)" y1="46" :x2="iterToX(15)" y2="312" stroke="#93c5fd" stroke-width="1.5" stroke-dasharray="3 3" />

        <!-- Stop line at iter 32 -->
        <line :x1="iterToX(32)" y1="46" :x2="iterToX(32)" y2="312" stroke="#b45309" stroke-width="2" stroke-dasharray="5 4" />
        <text :x="iterToX(32) + 8" y="72" font-size="12" font-weight="700" fill="#92400e">
          Stopped at iter 32
        </text>
        <text :x="iterToX(32) + 8" y="88" font-size="11.5" fill="#92400e">
          (time &amp; budget)
        </text>
        <text :x="iterToX(41)" y="196" text-anchor="middle" font-size="12" font-weight="600" fill="#b45309">
          No checkpoints run
        </text>
        <text :x="iterToX(41)" y="213" text-anchor="middle" font-size="11" fill="#b45309">
          (cont2 planned to iter 50)
        </text>

        <!-- Series 3 (optional): C4D 640 vs 1600 (iters 4, 8, 12) -->
        <g v-if="showC4dSeries && (seriesFilter === 'all' || seriesFilter === 'v5')">
          <polyline
            :points="polylineFor(C4D_POINTS)"
            fill="none"
            stroke="#7c3aed"
            stroke-width="2"
            stroke-dasharray="5 3"
          />
          <g
            v-for="pt in C4D_POINTS"
            :key="`c4d-${pt.iter}`"
            class="data-pt"
            @click="selectedKey = pt.key"
          >
            <polygon
              :points="diamondPoints(iterToX(pt.iter), scoreToY(pt.score), selectedKey === pt.key ? 8 : 6)"
              fill="#7c3aed"
              stroke="#ffffff"
              stroke-width="1.5"
            />
            <text
              :x="iterToX(pt.iter)"
              :y="scoreToY(pt.score) + 18"
              text-anchor="middle"
              font-size="11"
              font-weight="600"
              fill="#5b21b6"
              stroke="#ffffff"
              stroke-width="3"
              stroke-linejoin="round"
              paint-order="stroke"
            >
              {{ pt.score.toFixed(3) }}
            </text>
          </g>
        </g>

        <!-- Series 2: vs v6b Seed (iters 10, 15, 20, 25) -->
        <g v-if="seriesFilter === 'all' || seriesFilter === 'seed'">
          <polyline
            :points="polylineFor(SEED_POINTS)"
            fill="none"
            stroke="#059669"
            stroke-width="2.5"
          />
          <g
            v-for="pt in SEED_POINTS"
            :key="`seed-${pt.iter}`"
            class="data-pt"
            @click="selectedKey = pt.key"
          >
            <rect
              :x="iterToX(pt.iter) - (selectedKey === pt.key ? 6.5 : 5)"
              :y="scoreToY(pt.score) - (selectedKey === pt.key ? 6.5 : 5)"
              :width="selectedKey === pt.key ? 13 : 10"
              :height="selectedKey === pt.key ? 13 : 10"
              rx="2"
              fill="#059669"
              stroke="#ffffff"
              stroke-width="1.5"
            />
            <text
              :x="pt.iter === 15 ? iterToX(pt.iter) - 9 : iterToX(pt.iter)"
              :y="pt.iter === 10 ? scoreToY(pt.score) + 18 : scoreToY(pt.score) - 11"
              :text-anchor="pt.iter === 15 ? 'end' : 'middle'"
              font-size="11"
              font-weight="700"
              fill="#065f46"
              stroke="#ffffff"
              stroke-width="3"
              stroke-linejoin="round"
              paint-order="stroke"
            >
              {{ (pt.score * 100).toFixed(1) }}%
            </text>
          </g>
        </g>

        <!-- Series 1: vs v5 iter 144 at 270 vs 630 (iters 13, 16, 20, 24, 28, 32) -->
        <g v-if="seriesFilter === 'all' || seriesFilter === 'v5'">
          <polyline
            :points="polylineFor(ARM_POINTS)"
            fill="none"
            stroke="#2563eb"
            stroke-width="2.75"
          />
          <g
            v-for="pt in ARM_POINTS"
            :key="`arm-${pt.iter}`"
            class="data-pt"
            @click="selectedKey = pt.key"
          >
            <circle
              :cx="iterToX(pt.iter)"
              :cy="scoreToY(pt.score)"
              :r="selectedKey === pt.key ? 7.5 : 5.5"
              fill="#2563eb"
              stroke="#ffffff"
              stroke-width="1.5"
            />
            <text
              :x="pt.labelDx ? iterToX(pt.iter) + pt.labelDx : iterToX(pt.iter)"
              :y="scoreToY(pt.score) + (pt.labelDy ?? -10)"
              :text-anchor="pt.anchor ?? 'middle'"
              font-size="11"
              font-weight="700"
              fill="#1e40af"
              stroke="#ffffff"
              stroke-width="3"
              stroke-linejoin="round"
              paint-order="stroke"
            >
              {{ pt.score.toFixed(3) }}
            </text>
          </g>
        </g>

        <!-- Legend inside bottom-left/bottom row -->
        <g transform="translate(68, 368)">
          <circle cx="8" cy="0" r="5" fill="#2563eb" />
          <text x="19" y="4" font-size="11.5" fill="#334155">vs. v5 iter 144 (270 vs 630, 300 games)</text>

          <rect x="275" y="-5" width="10" height="10" rx="2" fill="#059669" />
          <text x="291" y="4" font-size="11.5" fill="#334155">vs. v6b seed (iter 0)</text>

          <polygon points="448,-6 454,0 448,6 442,0" fill="#7c3aed" />
          <text x="460" y="4" font-size="11.5" fill="#334155">vs. v5 iter 144 (640 vs 1,600, 200 games)</text>
        </g>
      </svg>

      <p class="source-note">
        Sources: <code>v6b-10x128.yaml</code>, <code>v6b-10x128-cont1.yaml</code>, <code>v6b-10x128-cont2.yaml</code>, and GitHub issue #49. Each 300-game step stays inside single-match sampling noise.
      </p>
    </div>
  </div>
</template>

<script setup>
import { computed, ref } from 'vue';

const SERIES_OPTIONS = [
  { id: 'all', label: 'Both Series' },
  { id: 'v5', label: 'vs. v5 iter 144 (270 vs 630)' },
  { id: 'seed', label: 'vs. v6b Seed (iter 0)' },
];

const ARM_POINTS = [
  { key: 'arm-13', iter: 13, score: 0.480, lr: '0.0003', seriesLabel: 'vs. v5 (270 vs 630)', detail: '300 games, 3 random opening plies (pre-LR drop)', labelDy: -8, labelDx: -8, anchor: 'end' },
  { key: 'arm-16', iter: 16, score: 0.460, lr: '0.0001', seriesLabel: 'vs. v5 (270 vs 630)', detail: '300 games, 3 random opening plies', labelDy: 17 },
  { key: 'arm-20', iter: 20, score: 0.480, lr: '0.0001', seriesLabel: 'vs. v5 (270 vs 630)', detail: '300 games, 3 random opening plies', labelDy: 17 },
  { key: 'arm-24', iter: 24, score: 0.497, lr: '0.0001', seriesLabel: 'vs. v5 (270 vs 630)', detail: '300 games, 3 random opening plies (peak measured)', labelDy: 22 },
  { key: 'arm-28', iter: 28, score: 0.490, lr: '0.00005', seriesLabel: 'vs. v5 (270 vs 630)', detail: '300 games, 3 random opening plies', labelDy: 17 },
  { key: 'arm-32', iter: 32, score: 0.487, lr: '0.00005', seriesLabel: 'vs. v5 (270 vs 630)', detail: '300 games, 3 random opening plies (run stopped here)', labelDy: 17, labelDx: -8, anchor: 'end' },
];

const SEED_POINTS = [
  { key: 'seed-10', iter: 10, score: 0.450, lr: '0.0003', seriesLabel: 'vs. v6b Seed (iter 0)', detail: 'seed check reported in v6b-10x128-cont2.yaml' },
  { key: 'seed-15', iter: 15, score: 0.500, lr: '0.0001', seriesLabel: 'vs. v6b Seed (iter 0)', detail: 'seed check reported in v6b-10x128-cont2.yaml' },
  { key: 'seed-20', iter: 20, score: 0.513, lr: '0.0001', seriesLabel: 'vs. v6b Seed (iter 0)', detail: 'seed check reported in v6b-10x128-cont2.yaml' },
  { key: 'seed-25', iter: 25, score: 0.525, lr: '0.00005', seriesLabel: 'vs. v6b Seed (iter 0)', detail: 'seed check reported in v6b-10x128-cont2.yaml' },
];

const C4D_POINTS = [
  { key: 'c4d-4', iter: 4, score: 0.465, lr: '0.0003', seriesLabel: 'vs. v5 (640 vs 1,600)', detail: '200 games, 3 random opening plies (2.5× C4D ratio)' },
  { key: 'c4d-8', iter: 8, score: 0.475, lr: '0.0003', seriesLabel: 'vs. v5 (640 vs 1,600)', detail: '200 games, 3 random opening plies (2.5× C4D ratio)' },
  { key: 'c4d-12', iter: 12, score: 0.460, lr: '0.0003', seriesLabel: 'vs. v5 (640 vs 1,600)', detail: '200 games, 3 random opening plies (2.5× C4D ratio)' },
];

const ALL_POINTS = [...ARM_POINTS, ...SEED_POINTS, ...C4D_POINTS];

const X_TICKS = [0, 5, 10, 15, 20, 25, 30, 32, 40, 50];
const Y_TICKS = [0.44, 0.46, 0.48, 0.50, 0.52, 0.54];

const seriesFilter = ref('all');
const showC4dSeries = ref(true);
const selectedKey = ref('arm-24');

const selectedPoint = computed(
  () => ALL_POINTS.find((p) => p.key === selectedKey.value) ?? ARM_POINTS[3]
);

const selectedBadgeClass = computed(() => {
  if (selectedPoint.value.key.startsWith('seed')) return 'badge-green';
  if (selectedPoint.value.key.startsWith('c4d')) return 'badge-purple';
  return 'badge-blue';
});

function iterToX(it) {
  return 68 + (it / 50) * 680;
}

function scoreToY(s) {
  return 312 - ((s - 0.43) / (0.55 - 0.43)) * 266;
}

function polylineFor(pts) {
  return pts.map((p) => `${iterToX(p.iter)},${scoreToY(p.score)}`).join(' ');
}

function diamondPoints(cx, cy, r) {
  return `${cx},${cy - r} ${cx + r},${cy} ${cx},${cy + r} ${cx - r},${cy}`;
}

function isHalf(v) {
  return Math.abs(v - 0.5) < 0.001;
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
  margin-bottom: 10px;
}

.btn-row {
  display: flex;
  flex-wrap: wrap;
  justify-content: center;
  gap: 8px;
}

.ctrl-btn {
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

.ctrl-btn:hover {
  background: #f1f5f9;
  border-color: #94a3b8;
}

.ctrl-btn.active {
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

.badge-blue {
  background: #dbeafe;
  color: #1e40af;
  border: 1px solid #93c5fd;
}

.badge-green {
  background: #dcfce7;
  color: #166534;
  border: 1px solid #86efac;
}

.badge-purple {
  background: #ede9fe;
  color: #5b21b6;
  border: 1px solid #c4b5fd;
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

.data-pt {
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
}
</style>
