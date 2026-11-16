<template>
  <div class="diagram-wrapper">
    <div class="container">
      <h1>Score, 95% Confidence Interval, and Implied Elo</h1>

      <div class="controls-panel">
        <div class="preset-row" role="group" aria-label="Evaluation presets">
          <button
            v-for="p in PRESETS"
            :key="p.id"
            type="button"
            class="preset-btn"
            :class="{ active: isPreset(p) }"
            @click="applyPreset(p)"
          >
            {{ p.label }}
          </button>
        </div>

        <div class="sliders-row">
          <label class="slider-item">
            <span>Score <code>s</code>: <strong>{{ score.toFixed(3) }}</strong></span>
            <input v-model.number="score" type="range" min="0.300" max="0.850" step="0.001" class="range-input" />
          </label>
          <label class="slider-item">
            <span>Games <code>N</code>: <strong>{{ games }}</strong></span>
            <input v-model.number="games" type="range" min="20" max="1000" step="10" class="range-input" />
          </label>
          <label class="slider-item">
            <span>Draws <code>D/N</code>: <strong>{{ (drawRate * 100).toFixed(0) }}%</strong></span>
            <input v-model.number="drawRate" type="range" min="0" max="0.20" step="0.05" class="range-input" />
          </label>
        </div>
      </div>

      <svg
        class="diagram-svg"
        viewBox="0 0 780 390"
        xmlns="http://www.w3.org/2000/svg"
        role="img"
        aria-label="Score confidence interval and logistic Elo conversion across sample sizes"
      >
        <!-- Top panel: Score confidence interval -->
        <text x="24" y="24" font-size="13.5" font-weight="700" fill="#0f172a">
          95% Confidence Interval on Score s = (W + D/2) / N
        </text>
        <text x="756" y="24" text-anchor="end" font-size="12" font-weight="700" :fill="ciCrossesHalf ? '#b45309' : '#15803d'">
          {{ ciCrossesHalf ? '95% CI crosses 0.500 (inconclusive)' : '95% CI clears 0.500' }}
        </text>

        <!-- Axis track -->
        <line x1="56" y1="78" x2="724" y2="78" stroke="#94a3b8" stroke-width="2" />
        <g v-for="t in SCORE_TICKS" :key="'st-' + t">
          <line :x1="scoreToX(t)" y1="72" :x2="scoreToX(t)" y2="84" stroke="#64748b" stroke-width="1.3" />
          <text :x="scoreToX(t)" y="101" text-anchor="middle" font-size="11.5" fill="#475569">
            {{ t.toFixed(2) }}
          </text>
        </g>

        <!-- 0.500 and 0.550 reference guides -->
        <line :x1="scoreToX(0.5)" y1="46" :x2="scoreToX(0.5)" y2="92" stroke="#475569" stroke-width="1.8" stroke-dasharray="4 3" />
        <text :x="scoreToX(0.5) - 6" y="42" text-anchor="end" font-size="11" font-weight="700" fill="#475569">
          0.50 even
        </text>

        <line :x1="scoreToX(0.55)" y1="46" :x2="scoreToX(0.55)" y2="92" stroke="#2563eb" stroke-width="1.6" stroke-dasharray="3 3" />
        <text :x="scoreToX(0.55) + 6" y="42" text-anchor="start" font-size="11" font-weight="700" fill="#2563eb">
          0.55 gate
        </text>

        <!-- 95% CI band -->
        <rect
          :x="scoreToX(clampAxisScore(ciLow))"
          y="69"
          :width="Math.max(2, scoreToX(clampAxisScore(ciHigh)) - scoreToX(clampAxisScore(ciLow)))"
          height="18"
          rx="4"
          :fill="ciCrossesHalf ? '#fde68a' : '#bbf7d0'"
          :stroke="ciCrossesHalf ? '#d97706' : '#16a34a'"
          stroke-width="1.8"
        />
        <!-- Point marker -->
        <circle
          :cx="scoreToX(clampAxisScore(score))"
          cy="78"
          r="6"
          :fill="ciCrossesHalf ? '#b45309' : '#15803d'"
          stroke="#ffffff"
          stroke-width="2"
        />
        <rect
          :x="clampLabelX(scoreToX(clampAxisScore(score))) - 48"
          y="47"
          width="96"
          height="16"
          rx="4"
          fill="#f8fafc"
          opacity="0.92"
        />
        <text
          :x="clampLabelX(scoreToX(clampAxisScore(score)))"
          y="59"
          text-anchor="middle"
          font-size="12"
          font-weight="700"
          fill="#0f172a"
        >
          {{ score.toFixed(3) }} ± {{ margin.toFixed(3) }}
        </text>

        <!-- Divider -->
        <line x1="20" y1="120" x2="760" y2="120" stroke="#e2e8f0" stroke-width="1.5" />

        <!-- Bottom Left: Logistic Elo curve -->
        <rect x="20" y="134" width="460" height="240" rx="8" fill="#ffffff" stroke="#cbd5e1" stroke-width="1.5" />
        <text x="36" y="156" font-size="13" font-weight="700" fill="#0f172a">
          Elo Curve: Δ = −400 log₁₀(1/s − 1)
        </text>

        <!-- Plot grid -->
        <g v-for="e in [-300, -150, 0, 150, 300]" :key="'eg-' + e">
          <line
            x1="78"
            :y1="eloToY(e)"
            x2="460"
            :y2="eloToY(e)"
            :stroke="e === 0 ? '#94a3b8' : '#f1f5f9'"
            :stroke-width="e === 0 ? '1.3' : '1'"
          />
          <text x="72" :y="eloToY(e) + 4" text-anchor="end" font-size="10.5" fill="#64748b">
            {{ e > 0 ? '+' + e : e }}
          </text>
        </g>

        <g v-for="sx in [0.2, 0.35, 0.5, 0.65, 0.8, 0.9]" :key="'sg-' + sx">
          <line
            :x1="curveX(sx)"
            y1="172"
            :x2="curveX(sx)"
            y2="342"
            :stroke="sx === 0.5 ? '#94a3b8' : '#f1f5f9'"
            :stroke-width="sx === 0.5 ? '1.3' : '1'"
          />
          <text :x="curveX(sx)" y="357" text-anchor="middle" font-size="10.5" fill="#64748b">
            {{ sx.toFixed(2) }}
          </text>
        </g>

        <!-- CI vertical band on curve -->
        <rect
          :x="curveX(clampCurveScore(ciLow))"
          y="172"
          :width="Math.max(2, curveX(clampCurveScore(ciHigh)) - curveX(clampCurveScore(ciLow)))"
          height="170"
          :fill="ciCrossesHalf ? '#fef3c7' : '#dcfce7'"
          opacity="0.7"
        />

        <!-- Curve path -->
        <path :d="eloCurvePath" fill="none" stroke="#2563eb" stroke-width="2.5" />

        <!-- Point projection lines -->
        <line
          :x1="curveX(clampCurveScore(score))"
          y1="342"
          :x2="curveX(clampCurveScore(score))"
          :y2="eloToY(clampPlotElo(elo))"
          stroke="#0f172a"
          stroke-width="1.3"
          stroke-dasharray="3 3"
        />
        <line
          x1="78"
          :y1="eloToY(clampPlotElo(elo))"
          :x2="curveX(clampCurveScore(score))"
          :y2="eloToY(clampPlotElo(elo))"
          stroke="#0f172a"
          stroke-width="1.3"
          stroke-dasharray="3 3"
        />
        <circle
          :cx="curveX(clampCurveScore(score))"
          :cy="eloToY(clampPlotElo(elo))"
          r="5.5"
          :fill="ciCrossesHalf ? '#b45309' : '#15803d'"
          stroke="#ffffff"
          stroke-width="1.5"
        />

        <!-- Bottom Right: Readout box -->
        <rect x="496" y="134" width="264" height="240" rx="8" fill="#ffffff" stroke="#cbd5e1" stroke-width="1.5" />
        <text x="512" y="158" font-size="13" font-weight="700" fill="#0f172a">
          Calculated Statistics
        </text>
        <text x="512" y="185" font-size="12.5" fill="#334155">
          Score s: <tspan font-weight="700">{{ score.toFixed(3) }}</tspan> (N = {{ games }})
        </text>
        <text x="512" y="209" font-size="12.5" fill="#334155">
          Std error SE: <tspan font-weight="700">{{ se.toFixed(4) }}</tspan>
        </text>
        <text x="512" y="233" font-size="12.5" fill="#334155">
          95% CI (score): <tspan font-weight="700">[{{ ciLow.toFixed(3) }}, {{ ciHigh.toFixed(3) }}]</tspan>
        </text>
        <text x="512" y="259" font-size="12.5" fill="#1d4ed8">
          Implied Elo: <tspan font-weight="700">{{ fmtElo(elo) }}</tspan>
        </text>
        <text x="512" y="283" font-size="12.5" fill="#1d4ed8">
          95% CI (Elo): <tspan font-weight="700">[{{ fmtElo(eloLow) }}, {{ fmtElo(eloHigh) }}]</tspan>
        </text>
        <line x1="512" y1="300" x2="744" y2="300" stroke="#e2e8f0" stroke-width="1.2" />
        <text x="512" y="322" font-size="12" font-weight="700" fill="#0f172a">
          Games for margin ≤ |s − 0.50|:
        </text>
        <text x="512" y="343" font-size="12" fill="#475569">
          <template v-if="Math.abs(score - 0.5) < 0.002">
            s = 0.500 (no difference to detect)
          </template>
          <template v-else>
            N ≈ <tspan font-weight="700" fill="#0f172a">{{ gamesNeeded }} games</tspan> (for 95% CI to clear 0.50)
          </template>
        </text>
        <text x="512" y="362" font-size="10.5" fill="#64748b">
          Matches eval_local.py SE &amp; Elo formula
        </text>
      </svg>
    </div>
  </div>
</template>

<script setup>
import { computed, ref } from 'vue';

const PRESETS = [
  { id: 'g40', label: '40 games · 0.550 (old gate)', score: 0.55, games: 40, drawRate: 0 },
  { id: 'g80', label: '80 games · 0.550 (cont4 gate)', score: 0.55, games: 80, drawRate: 0 },
  { id: 'g200', label: '200 games · 0.540 (v6b it12 equal searches)', score: 0.54, games: 200, drawRate: 0 },
  { id: 'g300', label: '300 games · 0.497 (v6b it24 equal time)', score: 0.497, games: 300, drawRate: 0 },
  { id: 'g900', label: '900 games · 0.550 (selection scale)', score: 0.55, games: 900, drawRate: 0 },
];

const SCORE_TICKS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];

const score = ref(0.55);
const games = ref(40);
const drawRate = ref(0);

function applyPreset(p) {
  score.value = p.score;
  games.value = p.games;
  drawRate.value = p.drawRate;
}

function isPreset(p) {
  return (
    Math.abs(score.value - p.score) < 0.001 &&
    games.value === p.games &&
    Math.abs(drawRate.value - p.drawRate) < 0.001
  );
}

const variance = computed(() => {
  const s = score.value;
  return Math.max(0, s * (1 - s) - 0.25 * drawRate.value);
});

const se = computed(() => Math.sqrt(variance.value / games.value));
const margin = computed(() => 1.96 * se.value);
const ciLow = computed(() => Math.max(0, score.value - margin.value));
const ciHigh = computed(() => Math.min(1, score.value + margin.value));
const ciCrossesHalf = computed(() => ciLow.value <= 0.5 && ciHigh.value >= 0.5);

function scoreToElo(s) {
  const clipped = Math.min(Math.max(s, 1e-3), 1 - 1e-3);
  return -400 * Math.log10(1 / clipped - 1);
}

const elo = computed(() => scoreToElo(score.value));
const eloLow = computed(() => scoreToElo(ciLow.value));
const eloHigh = computed(() => scoreToElo(ciHigh.value));

const gamesNeeded = computed(() => {
  const diff = Math.abs(score.value - 0.5);
  if (diff < 0.002) return 0;
  return Math.ceil((1.96 * Math.sqrt(variance.value) / diff) ** 2);
});

function scoreToX(s) {
  return 56 + ((s - 0.2) / 0.7) * 668;
}

function clampAxisScore(s) {
  return Math.min(0.9, Math.max(0.2, s));
}

function clampLabelX(x) {
  return Math.min(690, Math.max(90, x));
}

function curveX(s) {
  return 78 + ((s - 0.15) / 0.75) * 382;
}

function clampCurveScore(s) {
  return Math.min(0.9, Math.max(0.15, s));
}

function eloToY(e) {
  return 257 - (e / 350) * 85;
}

function clampPlotElo(e) {
  return Math.min(340, Math.max(-340, e));
}

const eloCurvePath = computed(() => {
  const pts = [];
  for (let i = 0; i <= 75; i++) {
    const s = 0.15 + (i / 75) * 0.75;
    const e = clampPlotElo(scoreToElo(s));
    pts.push(`${i === 0 ? 'M' : 'L'}${curveX(s).toFixed(1)},${eloToY(e).toFixed(1)}`);
  }
  return pts.join(' ');
});

function fmtElo(val) {
  const rounded = Math.round(val);
  return rounded >= 0 ? `+${rounded}` : `${rounded}`;
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
  gap: 8px;
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
