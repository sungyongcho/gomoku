<template>
  <div class="diagram-wrapper">
    <div class="container">
      <h1>Illustrative Gate Simulation: Colour Balance &amp; Sample Size</h1>

      <div class="controls-panel">
        <div class="btn-row" role="group" aria-label="Challenger base strength">
          <span class="group-label">Challenger:</span>
          <button
            v-for="opt in STRENGTH_OPTS"
            :key="opt.id"
            type="button"
            class="ctrl-btn"
            :class="{ active: strengthId === opt.id }"
            @click="strengthId = opt.id"
          >
            {{ opt.label }}
          </button>
        </div>

        <div class="btn-row" role="group" aria-label="Opening Black win rate">
          <span class="group-label">Opening:</span>
          <button
            v-for="opt in OPENING_OPTS"
            :key="opt.id"
            type="button"
            class="ctrl-btn"
            :class="{ active: openingId === opt.id }"
            @click="openingId = opt.id"
          >
            {{ opt.label }}
          </button>
        </div>

        <div class="btn-row" role="group" aria-label="Colour assignment and game count">
          <button
            v-for="opt in ALLOC_OPTS"
            :key="opt.id"
            type="button"
            class="ctrl-btn"
            :class="{ active: allocId === opt.id }"
            @click="allocId = opt.id"
          >
            {{ opt.label }}
          </button>

          <button
            type="button"
            class="ctrl-btn"
            :class="{ active: numGames === 40 }"
            @click="numGames = 40"
          >
            40 games (≥ 22)
          </button>
          <button
            type="button"
            class="ctrl-btn"
            :class="{ active: numGames === 80 }"
            @click="numGames = 80"
          >
            80 games (≥ 44)
          </button>

          <button type="button" class="reroll-btn" @click="simSeed += 1">
            Re-roll 200 gates
          </button>
        </div>
      </div>

      <svg
        class="diagram-svg"
        viewBox="0 0 780 390"
        xmlns="http://www.w3.org/2000/svg"
        role="img"
        aria-label="Illustrative promotion gate distribution showing exact and simulated pass rates"
      >
        <!-- Illustrative disclaimer -->
        <rect x="20" y="8" width="740" height="24" rx="6" fill="#fffbeb" stroke="#fcd34d" stroke-width="1.2" />
        <text x="390" y="24" text-anchor="middle" font-size="11.5" font-weight="600" fill="#92400e">
          Illustrative simulation — logistic win model with measured Black rates (82% at 4-ply, 38% at 3-ply); games independent
        </text>

        <!-- Summary boxes -->
        <rect x="20" y="42" width="236" height="68" rx="8" fill="#ffffff" stroke="#cbd5e1" stroke-width="1.5" />
        <text x="34" y="62" font-size="11.5" font-weight="700" fill="#475569">
          Exact Pass Rate (≥ {{ passThreshold }} / {{ numGames }})
        </text>
        <text x="34" y="88" font-size="22" font-weight="800" :fill="passHeadlineColor">
          {{ (exactPassRate * 100).toFixed(1) }}%
        </text>
        <text x="118" y="88" font-size="11.5" fill="#64748b">
          ({{ simPasses }} / 200 sim)
        </text>

        <rect x="272" y="42" width="236" height="68" rx="8" fill="#ffffff" stroke="#cbd5e1" stroke-width="1.5" />
        <text x="286" y="62" font-size="11.5" font-weight="700" fill="#475569">
          Expected Challenger Score
        </text>
        <text x="286" y="88" font-size="20" font-weight="800" fill="#0f172a">
          {{ (expectedScore * 100).toFixed(1) }}%
        </text>
        <text x="364" y="88" font-size="11.5" fill="#64748b">
          ({{ (expectedScore * numGames).toFixed(1) }} / {{ numGames }} wins)
        </text>

        <rect x="524" y="42" width="236" height="68" rx="8" fill="#ffffff" stroke="#cbd5e1" stroke-width="1.5" />
        <text x="538" y="62" font-size="11.5" font-weight="700" fill="#475569">
          Model Game Probabilities
        </text>
        <text x="538" y="83" font-size="12" fill="#1e293b">
          As Black: <tspan font-weight="700">{{ (pWinAsBlack * 100).toFixed(1) }}%</tspan>
          · White: <tspan font-weight="700">{{ (pWinAsWhite * 100).toFixed(1) }}%</tspan>
        </text>
        <text x="538" y="100" font-size="11" fill="#64748b">
          Split: {{ blackGames }}B / {{ numGames - blackGames }}W
        </text>

        <!-- Histogram card -->
        <rect x="20" y="122" width="740" height="254" rx="8" fill="#ffffff" stroke="#cbd5e1" stroke-width="1.5" />
        <text x="36" y="146" font-size="13" font-weight="700" fill="#0f172a">
          200 Simulated {{ numGames }}-Game Gates (Distribution of Challenger Wins)
        </text>
        <text x="744" y="146" text-anchor="end" font-size="11.5" font-weight="600" :fill="isFalsePositiveCase ? '#dc2626' : '#15803d'">
          {{ isFalsePositiveCase ? 'Shaded right of 55% = false promotion' : 'Shaded right of 55% = true promotion' }}
        </text>

        <!-- Pass region background -->
        <rect
          :x="histPassX"
          y="160"
          :width="Math.max(0, 738 - histPassX)"
          height="168"
          :fill="isFalsePositiveCase ? '#fee2e2' : '#dcfce7'"
          opacity="0.6"
        />

        <!-- Threshold line at 55% -->
        <line
          :x1="histPassX"
          y1="156"
          :x2="histPassX"
          y2="334"
          :stroke="isFalsePositiveCase ? '#dc2626' : '#15803d'"
          stroke-width="2"
          stroke-dasharray="4 3"
        />
        <text
          :x="histPassX + 6"
          y="172"
          font-size="11"
          font-weight="700"
          :fill="isFalsePositiveCase ? '#dc2626' : '#15803d'"
        >
          55% gate (≥ {{ passThreshold }})
        </text>

        <!-- Baseline axis -->
        <line x1="48" y1="328" x2="738" y2="328" stroke="#94a3b8" stroke-width="1.5" />

        <!-- Histogram bars -->
        <g v-for="b in histBins" :key="'bin-' + b.wins">
          <rect
            v-if="b.count > 0"
            :x="b.x"
            :y="328 - b.h"
            :width="b.w"
            :height="b.h"
            rx="2"
            :fill="b.passes ? (isFalsePositiveCase ? '#dc2626' : '#16a34a') : '#64748b'"
          />
          <text
            v-if="b.showLabel"
            :x="b.x + b.w / 2"
            y="346"
            text-anchor="middle"
            font-size="10.5"
            fill="#475569"
          >
            {{ b.wins }}
          </text>
        </g>

        <text x="393" y="366" text-anchor="middle" font-size="11.5" fill="#64748b">
          Challenger wins out of {{ numGames }} games (seed #{{ simSeed }})
        </text>
      </svg>
    </div>
  </div>
</template>

<script setup>
import { computed, ref } from 'vue';

const STRENGTH_OPTS = [
  { id: 'weaker', label: 'Weaker (45% base)', baseP: 0.45 },
  { id: 'equal', label: 'Equal (50% base)', baseP: 0.5 },
  { id: 'stronger', label: 'Stronger (55% / +35 Elo)', baseP: 0.55 },
];

const OPENING_OPTS = [
  { id: 'p4', label: '4-ply (82% Black)', blackP: 0.82 },
  { id: 'p3', label: '3-ply (38% Black)', blackP: 0.38 },
  { id: 'even', label: 'Even (50% Black)', blackP: 0.5 },
];

const ALLOC_OPTS = [
  { id: 'balanced', label: 'Balanced pairs (50/50)' },
  { id: 'unbalanced', label: 'Unbalanced (+4 Black)' },
];

const strengthId = ref('equal');
const openingId = ref('p4');
const allocId = ref('unbalanced');
const numGames = ref(40);
const simSeed = ref(1);

const baseP = computed(() => STRENGTH_OPTS.find((o) => o.id === strengthId.value)?.baseP ?? 0.5);
const blackP = computed(() => OPENING_OPTS.find((o) => o.id === openingId.value)?.blackP ?? 0.5);

function logit(p) {
  return Math.log(p / (1 - p));
}
function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

const pWinAsBlack = computed(() => sigmoid(logit(baseP.value) + logit(blackP.value)));
const pWinAsWhite = computed(() => sigmoid(logit(baseP.value) - logit(blackP.value)));

const blackGames = computed(() => {
  const half = numGames.value / 2;
  return allocId.value === 'balanced' ? half : half + 4;
});
const whiteGames = computed(() => numGames.value - blackGames.value);

const passThreshold = computed(() => Math.ceil(0.55 * numGames.value));

const expectedScore = computed(
  () => (blackGames.value * pWinAsBlack.value + whiteGames.value * pWinAsWhite.value) / numGames.value
);

// Exact Poisson-binomial pass probability across blackGames and whiteGames
const exactPassRate = computed(() => {
  const n = numGames.value;
  let dist = new Array(n + 1).fill(0);
  dist[0] = 1;
  const applyGame = (p) => {
    const next = new Array(n + 1).fill(0);
    for (let k = 0; k <= n; k++) {
      if (dist[k] === 0) continue;
      next[k] += dist[k] * (1 - p);
      if (k + 1 <= n) next[k + 1] += dist[k] * p;
    }
    dist = next;
  };
  for (let i = 0; i < blackGames.value; i++) applyGame(pWinAsBlack.value);
  for (let i = 0; i < whiteGames.value; i++) applyGame(pWinAsWhite.value);
  let sum = 0;
  for (let k = passThreshold.value; k <= n; k++) sum += dist[k];
  return sum;
});

const isFalsePositiveCase = computed(() => baseP.value <= 0.5);
const passHeadlineColor = computed(() => {
  if (isFalsePositiveCase.value) {
    return exactPassRate.value >= 0.2 ? '#dc2626' : '#b45309';
  }
  return exactPassRate.value >= 0.5 ? '#15803d' : '#b45309';
});

function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const simResults = computed(() => {
  const seedBase =
    simSeed.value * 9973 +
    numGames.value * 31 +
    Math.round(baseP.value * 100) * 17 +
    Math.round(blackP.value * 100) * 13 +
    (allocId.value === 'balanced' ? 1 : 7);
  const rng = mulberry32(seedBase);
  const counts = new Array(numGames.value + 1).fill(0);
  let passes = 0;
  for (let t = 0; t < 200; t++) {
    let wins = 0;
    for (let i = 0; i < blackGames.value; i++) {
      if (rng() < pWinAsBlack.value) wins += 1;
    }
    for (let i = 0; i < whiteGames.value; i++) {
      if (rng() < pWinAsWhite.value) wins += 1;
    }
    counts[wins] += 1;
    if (wins >= passThreshold.value) passes += 1;
  }
  return { counts, passes };
});

const simPasses = computed(() => simResults.value.passes);

const minBin = computed(() => (numGames.value === 40 ? 6 : 16));
const maxBin = computed(() => (numGames.value === 40 ? 34 : 64));

const histPassX = computed(() => {
  const span = maxBin.value - minBin.value + 1;
  const step = 690 / span;
  return 48 + (passThreshold.value - minBin.value) * step;
});

const histBins = computed(() => {
  const counts = simResults.value.counts;
  const lo = minBin.value;
  const hi = maxBin.value;
  const span = hi - lo + 1;
  const step = 690 / span;
  const maxC = Math.max(12, ...counts.slice(lo, hi + 1));
  const bins = [];
  for (let w = lo; w <= hi; w++) {
    const c = counts[w] || 0;
    bins.push({
      wins: w,
      count: c,
      x: 48 + (w - lo) * step + 1,
      w: Math.max(2, step - 2),
      h: (c / maxC) * 155,
      passes: w >= passThreshold.value,
      showLabel: numGames.value === 40 ? w % 4 === 0 : w % 8 === 0,
    });
  }
  return bins;
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
  gap: 8px;
  margin-bottom: 10px;
}

.btn-row {
  display: flex;
  flex-wrap: wrap;
  justify-content: center;
  align-items: center;
  gap: 7px;
}

.group-label {
  font-size: 12.5px;
  font-weight: 700;
  color: #475569;
  margin-right: 2px;
}

.ctrl-btn,
.reroll-btn {
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

.ctrl-btn:hover,
.reroll-btn:hover {
  background: #f1f5f9;
  border-color: #94a3b8;
}

.ctrl-btn.active {
  background: #eff6ff;
  border-color: #2563eb;
  color: #1e40af;
}

.reroll-btn {
  background: #fffbeb;
  border-color: #f59e0b;
  color: #92400e;
}

.reroll-btn:hover {
  background: #fef3c7;
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
